import json
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers import BertTokenizer
import ast


def DAE(bit_sequence, covertext, device, model, tokenizer, beam_num, no_repeat):
    LONG_BORING_TENNIS_ARTICLE = covertext.replace('\n', '')
    model.set_code(code=bit_sequence)
    article_input_ids = \
        tokenizer.batch_encode_plus([LONG_BORING_TENNIS_ARTICLE], return_tensors='pt', max_length=1024,
                                    truncation=True)[
            'input_ids'].to(device)
    model.set_text(text=article_input_ids[0])
    ids = model.generate(article_input_ids, num_beams=beam_num, length_penalty=1.0,
                         no_repeat_ngram_size=no_repeat, num_return_sequences=beam_num, return_dict_in_generate=True)
    id = ids['sequences']
    index = 0
    selected_sectence = tokenizer.decode(id[index].squeeze(), skip_special_tokens=True)
    selected_sectence = selected_sectence[0:selected_sectence.find("[unused10]")]
    return selected_sectence


def main(args):
    output_name = './datasets/{}/bart_bpw_{}_theta_{}'.format(args.dataset, args.bpw, args.theta, )
    model_name = args.pre_model_name
    device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained('model/user_bart/uer_bart_vocab.txt')
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    model.set_bpw(args.bpw)
    model.set_theta(args.theta)
    dataset_path = './datasets/{}/plaintext.txt'.format(args.dataset)
    print(dataset_path)
    data_p = open(f"{dataset_path}", "r", encoding='utf-8')
    covertexts = data_p.readlines()
    messages = []
    bit_path = './datasets/{}/message_bits.txt'.format(args.dataset)
    print(bit_path)
    fin = open(f"{bit_path}", "r")
    data = fin.readlines()
    for line in data:
        list_line = ast.literal_eval(line)
        messages.append(list_line)
    stega_info = []
    bpw_arr = []
    for i in tqdm(range(4)):
        bit_sequence = messages[i]
        model.set_total()
        covertext = covertexts[i]
        stega_text = DAE(bit_sequence, covertext, device, model, tokenizer, args.beam_num, args.no_repeat)
        total_step, bit_index = model.get_total()
        bpw_arr.append(bit_index[0] / total_step[0])
        stega_info.append(stega_text)
    results = {
        "bpw": bpw_arr,
        "covertexts": stega_info
    }
    print(output_name)
    with open(output_name, "w", encoding="utf8") as fout:
        json.dump(results, fout, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)


if __name__ == "__main__":
    psr = ArgumentParser()
    psr.add_argument("--dataset", type=str, default="shopping", choices=["ding", "shopping", ])
    psr.add_argument("--beam_num", type=int, default=1)
    psr.add_argument("--no_repeat", type=int, default=5)
    psr.add_argument("--model_name", type=str, default="bart-base-chinese")
    psr.add_argument("--pre_model_name", type=str, default="./model/user_bart")
    psr.add_argument("--device", type=int, default=0)
    psr.add_argument("--bpw", type=int, default=1)
    psr.add_argument("--theta", type=float, default="6.0")
    psr.add_argument("--score_method", type=str, default="avg")
    args = psr.parse_args()
    print(args)
    main(args)
