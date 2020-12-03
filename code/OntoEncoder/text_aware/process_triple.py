import os
import codecs


if __name__ == "__main__":

    datadir = '../../data'

    # dataset = 'AwA'
    dataset = 'ImageNet/ImNet_A'

    DATASET_DIR = os.path.join(datadir, dataset)
    DATA_DIR = os.path.join(datadir, dataset, 'onto_file')

    if dataset == 'NELL' or dataset == 'Wiki':
        triples_file = os.path.join(DATA_DIR, 'all_triples.txt')
        save_file = os.path.join(DATA_DIR, 'all_triples_htr.txt')
    else:
        triples_file = os.path.join(DATA_DIR, 'all_triples_names.txt')
        save_file = os.path.join(DATA_DIR, 'all_triples_names_htr.txt')

    wr_fp = open(save_file, 'w')
    text_file = codecs.open(triples_file, "r", "utf-8")
    lines = text_file.readlines()
    for line in lines:
        line_arr = line.rstrip("\r\n").split("\t")
        head = line_arr[0]
        rel = line_arr[1]
        tail = line_arr[2]

        wr_fp.write('%s\t%s\t%s\n' % (head, tail, rel))

    wr_fp.close()







