import sys
import re

val = sys.argv
dic = {}
diff_dic_pack = {}
diff_dic_ver = {}
path = ''


if len(val) == 3:
    path = './requirements_diff_out.txt'
elif len(val) == 4:
    path = val[3]
else:
    print("Usage : $ python extract_requirements.py [file1] [file2] ([outfile])\n")
    sys.exit(0)

if val[1] == val[2]:
    print('Same file\n')
    sys.exit(0)

try:
    with open(val[1]) as f:
        for s_line in f:
            if '#' in s_line[0]:
                continue
            s = re.split(r'([<>=~!\s])', s_line)
            li = ''
            for k in range(1, len(s)):
                if s[k] in ['\n', '\t', ' ', '']:
                    continue
                li = li + s[k]
            temp = {s[0]:li}
            dic = {**dic, **temp}

    with open(val[2]) as f:
        for s_line in f:
            if '#' == s_line[0]:
                continue
            s = re.split(r'([<>=~!\s])', s_line)
            li = ''
            for k in range(1, len(s)):
                if s[k] in ['\n', '\t', ' ', '']:
                    continue
                li = li + s[k]
            if s[0] in dic.keys():
                if li != dic[s[0]]:
                    temp = {s[0]:[dic[s[0]], li]}
                    diff_dic_ver = {**diff_dic_ver, **temp}
            else:
                temp = {s[0]:li}
                diff_dic_pack = {**diff_dic_pack, **temp}

    with open(path, mode='w+') as f:
        line = '------------package difference------------\n\n'
        f.write(line)
        keys = list(diff_dic_pack.keys())
        for i in range(len(diff_dic_pack)):
            line = f'{keys[i]}{diff_dic_pack[keys[i]]}\n'
            f.write(line)
        
        line = '\n\n------------version difference------------\n\n'
        f.write(line)
        keys = list(diff_dic_ver.keys())
        for i in range(len(diff_dic_ver)):
            line = keys[i].ljust(20) + ' :  ' + diff_dic_ver[keys[i]][0].ljust(15) + ' -->  ' + diff_dic_ver[keys[i]][1] + '\n'
            f.write(line)
    print(f'Succeed. See {path}')

except Exception as e:
    print(f'Error: {e}')
