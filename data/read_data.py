

def read_doc_data(fn, col_delim="/", word_delim=" ", sent_delim="。", w2c_doc_limit=None, w2c_target_limit=None):
    with open(fn, "r", encoding="utf-8") as f:
        lines = f.readlines()
    rv = []
    w2c_doc, w2c_target = {}, {}
    for line in lines:
        line = line.rstrip()
        if len(line) == 0: continue
        cols = line.split(col_delim)
        target_seg_list = cols[0].split(word_delim)
        # target_seg_list = cols[1].split(word_delim)
        doc_seg_list = cols[1].split(word_delim)
        # doc_seg_list = cols[2].split(word_delim)
        doc_kw_seg_list = cols[2].split(word_delim)
        # doc_kw_seg_list = cols[4].split(word_delim)
        doc_sents_seg_list = [ [w for w in sent_str.split(word_delim) if len(w) > 0]
                               for sent_str in cols[1].split(sent_delim) if len(sent_str.rstrip()) > 0]
        # doc_sents_seg_list = [ [w for w in sent_str.split(word_delim) if len(w) > 0]
        #                        for sent_str in cols[2].split(sent_delim) if len(sent_str.rstrip()) > 0]
        for word in doc_seg_list:
            if word not in w2c_doc: w2c_doc[word] = 0
            w2c_doc[word] += 1
        for word in target_seg_list:
            if word not in w2c_target: w2c_target[word] = 0
            w2c_target[word] += 1
        rv.append([target_seg_list, doc_seg_list, doc_kw_seg_list, doc_sents_seg_list])
    if w2c_doc_limit is not None and len(w2c_doc) > w2c_doc_limit:
        tmp = sorted([(w,c) for w,c in w2c_doc.items()],key=lambda t:t[1],reverse=True)[:w2c_doc_limit]
        w2c_doc = {t[0]:t[1] for t in tmp}
    if w2c_target_limit is not None and len(w2c_target) > w2c_target_limit:
        tmp = sorted([(w,c) for w,c in w2c_target.items()],key=lambda t:t[1],reverse=True)[:w2c_target_limit]
        w2c_target = {t[0]: t[1] for t in tmp}
    return rv, w2c_doc, w2c_target
