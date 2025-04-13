# Two speakers situation
def split_and_pad_tensor_by_two(original_tensor):
    device = original_tensor.device  # 获取原始张量所在的设备
    batchsize, max_length = original_tensor.shape
    max_length_before_two = 0
    max_length_after_two = 0

    split_before_two = []
    split_after_two = []

    for row in original_tensor:
        two_index = (row == 2).nonzero(as_tuple=True)[0]
        if two_index.nelement() == 0:
            continue

        two_index = two_index.item()
        before_two = row[:two_index][row[:two_index] != -1]
        after_two = row[two_index+1:][row[two_index+1:] != -1]

        max_length_before_two = max(max_length_before_two, len(before_two))
        max_length_after_two = max(max_length_after_two, len(after_two))

        split_before_two.append(before_two)
        split_after_two.append(after_two)

    padded_before_two = [torch.cat((tensor, torch.full((max_length_before_two - len(tensor),), -1, device=device))) for tensor in split_before_two]
    padded_after_two = [torch.cat((tensor, torch.full((max_length_after_two - len(tensor),), -1, device=device))) for tensor in split_after_two]

    return torch.stack(padded_before_two).to(device), torch.stack(padded_after_two).to(device)


def count_negative_ones_per_row(tensor):
    return (tensor == -1).sum(dim=1).tolist()


# Three speakers situation
def split_tensor_by_two(original_tensor):
    device = original_tensor.device  # 获取原始张量所在的设备
    batchsize, max_length = original_tensor.shape

    # 分割后的数组长度
    max_lengths = [0, 0, 0]

    # 存储分割后的数组
    split_tensors = [[], [], []]

    for row in original_tensor:
        two_indices = (row == 2).nonzero(as_tuple=True)[0]
        if two_indices.nelement() < 2:
            continue  # 如果少于两个2，跳过这行

        # 获取两个2的索引
        first_two, second_two = two_indices[0].item(), two_indices[1].item()

        # 分割为三个部分
        before_first_two = row[:first_two][row[:first_two] != -1]
        between_twos = row[first_two + 1:second_two][row[first_two + 1:second_two] != -1]
        after_second_two = row[second_two + 1:][row[second_two + 1:] != -1]

        # 更新最大长度
        max_lengths[0] = max(max_lengths[0], len(before_first_two))
        max_lengths[1] = max(max_lengths[1], len(between_twos))
        max_lengths[2] = max(max_lengths[2], len(after_second_two))

        # 添加到列表
        split_tensors[0].append(before_first_two)
        split_tensors[1].append(between_twos)
        split_tensors[2].append(after_second_two)

    # 填充以保持统一长度
    for i in range(3):
        split_tensors[i] = [torch.cat((tensor, torch.full((max_lengths[i] - len(tensor),), -1, device=device))) for tensor in split_tensors[i]]
        split_tensors[i] = torch.stack(split_tensors[i]).to(device)

    return tuple(split_tensors)


def count_negative_ones_per_row(tensor):
    return (tensor == -1).sum(dim=1).tolist()


# ----------> CTC for two speakers
    def _calc_ctc_loss(
        self,
        encoder_out_spk1: torch.Tensor,
        encoder_out_spk2: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_spk1_pad: torch.Tensor,
        ys_spk1_pad_lens: torch.Tensor,
        ys_spk2_pad: torch.Tensor,
        ys_spk2_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc1 = self.ctc_spk1(encoder_out_spk1, encoder_out_lens, ys_spk1_pad, ys_spk1_pad_lens)
        loss_ctc2 = self.ctc_spk2(encoder_out_spk2, encoder_out_lens, ys_spk2_pad, ys_spk2_pad_lens)
        loss_ctc = (loss_ctc1 + loss_ctc2) / 2

        # Calc CER using CTC
        cer_ctc = None
        """
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        """
        return loss_ctc, cer_ctc



