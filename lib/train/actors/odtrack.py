from . import BaseActor
from lib.utils.misc import NestedTensor, interpolate
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class ODTrackActor(BaseActor):
    """ Actor for training ODTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)

        box_mask_z = []
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            for i in range(self.settings.num_template):
                box_mask_z.append(generate_mask_cond(self.cfg, template_list[i].shape[0], template_list[i].device,
                                                     data['template_anno'][i]))
            box_mask_z = torch.cat(box_mask_z, dim=1)

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                            total_epochs=ce_start_epoch + ce_warm_epoch,
                                            ITERS_PER_EPOCH=1,
                                            base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        out_dict = self.net(template=template_list,
                            search=search_img,  # search_list
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # currently only support the type of pred_dict is list
        # assert isinstance(pred_dict, list) predict is a tensor
        loss_dict = {}

        # generate gt gaussian map
        gt_gaussian_maps_list = generate_heatmap(gt_dict['search_anno'][0], self.cfg.DATA.SEARCH.SIZE,
                                                 self.cfg.MODEL.BACKBONE.STRIDE)

        # get GT
        gt_bbox = gt_dict['search_anno'][0]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = gt_gaussian_maps_list.unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")

        # for classification
        gt_labels = gt_dict['label'][0].view(-1)
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)
        # check if non-exist target object, set bounding box as [0, 0, 0, 0]
        visiable_index = []
        for index, value in enumerate(gt_labels.view(-1)):
            if value == 0:
                visiable_index.append(index)

        pred_boxes_vec = pred_boxes_vec.clone()
        gt_boxes_vec = gt_boxes_vec.clone()
        pred_boxes_vec[visiable_index, :] = torch.zeros(4).cuda()
        gt_boxes_vec[visiable_index, :] = torch.zeros(4).cuda()

        # only calculate giou loss, focal loss and l1 loss with image that contains target object
        # compute giou and iou for each sample in batch
        giou_loss = torch.zeros(pred_boxes_vec.size(0)).cuda()
        iou = torch.zeros(pred_boxes_vec.size(0)).cuda()
        l1_loss = torch.zeros(pred_boxes_vec.size(0)).cuda()
        focal_loss = torch.zeros(pred_boxes_vec.size(0)).cuda()

        for i, value in enumerate(gt_labels.view(-1)):
            if value != 0:
                try:
                    giou_loss[i], iou[i] = self.objective['giou'](pred_boxes_vec[i].view(1, -1), gt_boxes_vec[i].view(1, -1))  # (BN,4) (BN,4)
                except:
                    giou_loss[i], iou[i] = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                l1_loss[i] = self.objective['l1'](pred_boxes_vec[i].view(1, -1), gt_boxes_vec[i].view(1, -1))  # (BN,4) (BN,4)
                if 'score_map' in pred_dict:
                    focal_loss[i] = self.objective['focal'](pred_dict['score_map'][i], gt_gaussian_maps[i])

        loss_dict['giou'] = giou_loss.mean()
        loss_dict['l1'] = l1_loss.mean()
        loss_dict['focal'] = focal_loss.mean()

        # compute classification loss
        cls_loss = self.objective['cls'](pred_dict["pred_logits"].view(-1), gt_labels)
        loss_dict['cls'] = cls_loss

        # weighted sum
        loss = sum(loss_dict[k] * self.loss_weight[k] for k in loss_dict.keys() if k in self.loss_weight)

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {f"Loss/total": loss.item(),
                      f"Loss/giou": loss_dict['giou'].item(),
                      f"Loss/l1": loss_dict['l1'].item(),
                      f"Loss/location": loss_dict['focal'].item(),
                      f"Loss/cls": loss_dict['cls'].item(),
                      f"IoU": mean_iou.item()}

        if return_status:
            return loss, status
        else:
            return loss
