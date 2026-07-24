import sys
import random


class PreProcess:
    def __init__(self, conf_file='', is_test=False):
        self.conf = {}
        self.is_test = is_test
        self.sample_rate = 0.0
        self.need_normalize = False
        self.need_filt_slot = False
        self.filt_slot_set = set()
        self.zs_dic = {}

        if conf_file != '':
            self.load_conf(conf_file)
            self.init_conf()

    def g(self, ss):
        if ss in self.conf:
            return self.conf[ss]
        return ''

    def neg_samplling(self, rate):
        r = random.random()
        if r <= rate:
            return True
        return False

    def load_filtslot(slef, fpath):
        slot_dic = set()
        f = open(fpath)
        for line in f:
            line = line.strip()
            if line == '':
                continue
            lis = line.split(':')
            slot = int(lis[0])
            if len(lis) > 1:
                span = int(lis[1])
                for i in xrange(span):
                    si = slot + i
                    slot_dic.add(si)
            else:
                slot_dic.add(slot)
        f.close()
        return slot_dic

    def load_zscore(self, fpath):
        zs_dic = {}
        f = open(fpath)
        for line in f:
            line = line.strip()
            if line == '':
                continue
            lis = line.split(':')
            if len(lis) != 3:
                continue
            slot = int(lis[0])
            zs_dic[slot] = (float(lis[1]), float(lis[2]))
        f.close()
        return zs_dic

    def normal(self, slot, val, zs_dic):
        res = val
        if slot in zs_dic:
            zs = zs_dic[slot]
            mean = zs[0]
            std = zs[1]
            res = (val - mean) / std
        return res

    def load_conf(self, conf_file):
        self.conf = {}
        f = open(conf_file)
        for line in f:
            line = line.strip()
            if line == '':
                continue
            if line[0] == '#':
                continue
            lis = line.split('=')
            if len(lis) != 2:
                continue
            self.conf[lis[0]] = lis[1]
        f.close()

    def init_conf(self):
        self.sample_rate = 0.0
        if self.g('enable_sample') == '1':
            sample_rate = float(self.g('sample_rate'))
            self.sample_rate = sample_rate
        else:
            sys.stderr.write('sample_mod config fail\n')
            exit(1)

        self.need_filt_slot = False
        self.need_normalize = False

        self.filt_slot_set = set()
        self.zs_dic = {}

        if self.g('enable_fea_filt') == "1" and self.g('fea_filt_slot_conf') != "":
            self.filt_slot_set = self.load_filtslot(self.g('fea_filt_slot_conf'))
            if len(self.filt_slot_set) > 0:
                self.need_filt_slot = True
        if self.g('enable_zscore') == "1" and self.g('zscore_file') != "":
            self.zs_dic = self.load_zscore(self.g('zscore_file'))
            if len(self.zs_dic) > 0:
                self.need_normalize = True

    def parse_line_org(self, line):
        line = line.split('#', 1)[0].strip()
        if line == '':
            return
        lis = line.split(' ')
        label = lis[0]
        if self.is_test == False and label == '0' and self.g('enable_sample') == '1':
            if self.neg_samplling(self.sample_rate) == False:
                return
        if self.need_filt_slot == False and self.need_normalize == False:
            print(line)
            return
        out_lis = []
        out_lis.append(label)
        for i in xrange(1, len(lis)):
            li = lis[i].split(':')
            if len(li) != 2:
                out_lis.append(lis[i])
                continue
            slot = int(li[0])
            value = float(li[1])
            if self.need_normalize:
                val = self.normal(slot, value, self.zs_dic)
            if self.need_filt_slot:
                if slot in self.filt_slot_set:
                    val = 0.0
                    continue

            out_lis.append(str(slot) + ':' + str(val))
        print(' '.join(out_lis))

    def parse_line(self, line0):
        splits = line0.split('#', 1)
        if len(splits) < 2:
            return 
        line = splits[0].strip()
        add_i = splits[1].strip()
        if line == '':
            return
        lis = line.split(' ')
        label = lis[0]
        add_info = splits[1].strip().split('\t')

        is_cat_label = add_info[9]
        is_clk_label = add_info[8]
        uid = add_info[5]
        rec_index = int(add_info[3])
        u_nearby_imp_cnt = int(add_info[13])
        is_ext_cov = int(add_info[15])
        trace_imp_cnt = int(add_info[18])
        is_ext_clk = int(add_info[19])
        is_ext_cat = int(add_info[20])


        if self.is_test == True:
            return str(line) + "#" + str(add_i)
    #        return

        u_req_time = add_info[21]
        if u_req_time is not None and u_req_time != "\\N":
            u_req_time = int(u_req_time)
        else:
            u_req_time = 0

        pay_timestamp = add_info[22]
        if pay_timestamp is not None and pay_timestamp != "\\N":
            pay_timestamp = int(pay_timestamp)
        else:
            pay_timestamp = 0

        module = add_info[23]

        diff_time = 0
        if pay_timestamp > 0 and u_req_time > 0:
            diff_time = pay_timestamp - u_req_time

        reserve_flag = 0
        # add_list = add_info[:8]
        add_list = add_info[:15]

        if u_nearby_imp_cnt > 1000 and uid != '0':
            return
        if label == '0' and trace_imp_cnt <= 2 and is_clk_label == '0':
            if self.neg_samplling(0.5) == False:
                return

        is_ext_label = '0'
        if is_ext_cov == 1 and diff_time >= 0:
            is_ext_label = '1'

        add_list.append(is_ext_label)
        add_list.extend(add_info[16:])

        add_str = ""
        for info in add_list:
            add_str += info + '\t'
        add_i = add_str.strip()

        is_pos = 0
        if label != '0' or is_cat_label != '0' or is_clk_label != '0' or is_ext_label != '0':
            is_pos = 1

        if is_pos == 0 and self.g('enable_sample') == '1' and reserve_flag == 0:
            sample_rate = self.sample_rate
            if self.neg_samplling(sample_rate) == False:
                return

        if self.need_filt_slot == False and self.need_normalize == False:
            return line + '#' + add_i
        out_lis = []
        out_lis.append(label)
        for i in range(1, len(lis)):
            li = lis[i].split(':')
            if len(li) != 2:
                out_lis.append(lis[i])
                continue
            slot = int(li[0])
            value = float(li[1])
            if self.need_normalize:
                val = self.normal(slot, value, self.zs_dic)
            if self.need_filt_slot:
                if slot in self.filt_slot_set:
                    val = 0.0
                    continue

            out_lis.append(str(slot) + ':' + str(val))
        value = ' '.join(out_lis) + '#' + add_i
        return value
        #print(value)


# main
if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.stderr.write('input at least conf_file')
        exit(1)
    conf_file = sys.argv[1]

    is_test = False
    if len(sys.argv) > 2:
        if sys.argv[2] == 'test':
            is_test = True
    pre = PreProcess(conf_file=conf_file, is_test=is_test)
    for line in sys.stdin:
        pre.parse_line(line)