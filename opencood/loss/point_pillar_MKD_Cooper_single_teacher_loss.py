
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.loss.point_pillar_loss import PointPillarLoss

class PointPillarMKDCooperSingleTeacherLoss(PointPillarLoss):
    def __init__(self, args):
        super(PointPillarMKDCooperSingleTeacherLoss, self).__init__(args)
        self.kd = args['kd']

    def forward(self, output_dict, target_dict):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        total_loss = super().forward(output_dict, target_dict)

        ########## KL loss ############
        rm = output_dict['reg_preds']  # [B, 14, 50, 176]
        psm = output_dict['cls_preds'] # [B, 2, 50, 176]
        feature = output_dict['feature']

        teacher_rm = output_dict['teacher_reg_preds']
        teather_psm = output_dict['teacher_cls_preds']
        

        
        feature = output_dict['feature']
        teacher_feature = output_dict['teacher_feature']
        kl_loss_mean = nn.KLDivLoss(size_average=True, reduce=True)

        
        teacher_feature_1, teacher_feature_2, teacher_feature_3 = torch.chunk(teacher_feature, 3, 1)
        student_feature_1, student_feature_2, student_feature_3 = torch.chunk(feature, 3, 1)
        
        N1, C1, H1, W1 = teacher_feature_1.shape
        teacher_feature_1 = teacher_feature_1.permute(0, 2, 3, 1).reshape(N1 * H1 * W1, C1)
        student_feature_1 = student_feature_1.permute(0, 2, 3, 1).reshape(N1 * H1 * W1, C1)
        kd_loss_feature_1 = kl_loss_mean(
            F.log_softmax(student_feature_1, dim=1), F.softmax(teacher_feature_1, dim=1)
        )
        N2, C2, H2, W2 = teacher_feature_2.shape
        teacher_feature_2 = teacher_feature_2.permute(0, 2, 3, 1).reshape(N2 * H2 * W2, C2)
        student_feature_2 = student_feature_2.permute(0, 2, 3, 1).reshape(N2 * H2 * W2, C2)
        kd_loss_feature_2 = kl_loss_mean(
            F.log_softmax(student_feature_2, dim=1), F.softmax(teacher_feature_2, dim=1)
        )
        N3, C3, H3, W3 = teacher_feature_3.shape
        teacher_feature_3 = teacher_feature_3.permute(0, 2, 3, 1).reshape(N3 * H3 * W3, C3)
        student_feature_3 = student_feature_3.permute(0, 2, 3, 1).reshape(N3 * H3 * W3, C3)
        kd_loss_feature_3 = kl_loss_mean(
            F.log_softmax(student_feature_3, dim=1), F.softmax(teacher_feature_3, dim=1)
        )
        kd_loss = 0.2 * kd_loss_feature_1 + 0.3 * kd_loss_feature_2 + 0.5 * kd_loss_feature_3



        if self.kd.get('decoder_kd', False):
            N, C, H, W = teacher_rm.shape
            teacher_rm = teacher_rm.permute(0,2,3,1).reshape(N*H*W, C)
            student_rm = rm.permuate(0,2,3,1).reshape(N*H*W, C)
            kd_loss_rm = kl_loss_mean(
                    F.log_softmax(student_rm, dim=1), F.softmax(teacher_rm, dim=1)
                )

            N, C, H, W = teacher_psm.shape
            teacher_psm = teather_psm.permute(0,2,3,1).reshape(N*H*W, C)
            student_psm = psm.permuate(0,2,3,1).reshape(N*H*W, C)
            kd_loss_psm = kl_loss_mean(
                    F.log_softmax(student_psm, dim=1), F.softmax(teacher_psm, dim=1)
                )

            kd_loss += kd_loss_rm + kd_loss_psm

        kd_loss *= self.kd['weight']
        total_loss += kd_loss
        self.loss_dict.update({'total_loss': total_loss.item(),
                              'kd_loss': kd_loss.item()})


        return total_loss


    def logging(self, epoch, batch_id, batch_len, writer = None, suffix=''):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        dir_loss = self.loss_dict.get('dir_loss', 0)
        iou_loss = self.loss_dict.get('iou_loss', 0)
        kd_loss = self.loss_dict.get('kd_loss', 0)


        print("[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f || KD Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss, dir_loss, iou_loss, kd_loss))

        if not writer is None:
            writer.add_scalar('Regression_loss'+suffix, reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss'+suffix, cls_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Dir_loss'+suffix, dir_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Iou_loss'+suffix, iou_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Kd_loss'+suffix, kd_loss,
                            epoch*batch_len + batch_id)
