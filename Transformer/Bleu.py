"""
Author: Yanting Miao
"""
import torch
from torchnlp.metrics import get_moses_multi_bleu

def translation(output, trg_number2token, symbols):
    translate = [None] * output.size(0)
    for i in range(len(translate)):
        translate_out = ''
        for j in range(len(output[i])):
            idx = output[i][j].item()
            if trg_number2token[idx] != '<pad>':
                translate_out += trg_number2token[idx]
                if j < len(output[i]) - 1 and trg_number2token[output[i][j + 1].item()] not in symbols:
                    translate_out += ' '
        translate[i] = translate_out[:-1]

    return translate

def reconstruct_raw_trg(trg, symbols):
    reconstruct = [None] * len(trg)
    for i in range(len(trg)):
        line = ''
        for j in range(len(trg[i])):
            line += trg[i][j]
            if j < len(trg[i]) - 1 and trg[i][j + 1] not in symbols:
                line += ' '
        reconstruct[i] = line
    return reconstruct

def generate_translation(model, dataloader, device, trg_num2token, symbols):
    translate = []
    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)
        output = model(src, trg)
        output = torch.argmax(output, dim=1)
        output = translation(output, trg_num2token, symbols)
        translate.extend(output)
    return translate


if __name__ == '__main__':
    symbols = [',', '.', '?', '-', '!', '„']
    dev_raw_data = torch.load('dev_raw_trg.pt')
    test_raw_data = torch.load('test_raw_trg.pt')
    dev_dataloader = torch.load('dev_loader.pt')
    test_dataloader = torch.load('test_loader.pt')
    trg_token2num = torch.load('trg_token2num.pt')
    trg_num2token = dict((v, k) for k, v in trg_token2num.items())
    dev_raw_data = reconstruct_raw_trg(dev_raw_data, symbols)
    test_raw_data = reconstruct_raw_trg(test_raw_data, symbols)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('best_adam_transformer.pt').to(device)
    dev_translate = generate_translation(model, dev_dataloader, device, trg_num2token, symbols)
    bleu_score = get_moses_multi_bleu(dev_translate, dev_raw_data)
    print('Bleu score in dev dataset: %.2f' % (bleu_score))
    test_translate = generate_translation(model, test_dataloader, device, trg_num2token, symbols)
    bleu_score = get_moses_multi_bleu(test_translate, test_raw_data)
    print('Bleu score in test dataset: %.2f' % (bleu_score))