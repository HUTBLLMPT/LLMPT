import gzip#用于处理压缩文件的模块。
import html#用于处理HTML标签编码和解码的模块。
import os
from functools import lru_cache#用于实现最近最少使用缓存机制的装饰器函数。

import ftfy#用于处理乱码文本的。它提供了一些函数，可以自动检测和修复编码问题，使得文本能够正确显示和处理。
import regex as re#regex是Python中的一个替代标准正则表达替换式模块re的第三方模块。它提供了更强大的正则表达式功能和更好的性能。


@lru_cache()
def default_bpe():#os.path.dirname()函数获取当前文件的目录路径，再使用os.path.abspath()函数获取该目录路径的绝对路径，然后与"bpe_simple_vocab_16e6.txt.gz"拼接，最终返回拼接得到的文件路径字符串。
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    提供一个工具，用于将UTF-8字节表示的文本转换为对应的Unicode字符串。
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))#生成一个包含范围内所有ASCII可打印字符的列表。第一个生产生成一个包含从"!"到"~"的所有字符的列表。其余类似
    cs = bs[:]#cs是bs的副本
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)#通过2**8+n来保证新添加的值与已存在的值不重复。
            n += 1
    cs = [chr(n) for n in cs]#将列表cs中的所有值转换为对应的Unicode字符，生成一个由Unicode字符组成的列表，并将其赋值回变量cs
    return dict(zip(bs, cs))#逐一配对，然后生成字典，键是bs，值是cs


def get_pairs(word):
    """
    返回单词中的一组符号对。
    单词被表示为符号元组（符号是可变长度的字符串）。
    每一个都是其前一个和现在这个的组合。
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):#对文本进行基本的清洗和处理。
    text = ftfy.fix_text(text)#函数对文本进行修复，以处理可能出现的编码问题和Unicode字符转换。该函数的作用是尝试将混乱的或不正确编码的文本修复为正确的形式。
    text = html.unescape(html.unescape(text))#对文本进行HTML标签解码，将HTML实体转换回相应的字符形式。对文本进行两次解码操作的原因是为了确保解码所有的HTML实体，避免出现编码残留。HTML 实体字符是一些特殊字符或标记，以实体编码形式出现在 HTML 文档中。例如，&lt; 表示小于号 "<"，&gt; 表示大于号 ">"，&amp; 表示 & 符号等。
    return text.strip()#去除文本首尾空白字符


def whitespace_clean(text):#消除文本中空白文字
    text = re.sub(r'\s+', ' ', text)#使用re.sub()函数和正则表达式'\s+'将文本中连续的多个空白字符（包括空格、制表符、换行符等）替换为单个空格字符。这个操作可以将连续的多个空白字符合并为一个空格。
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):#传入路径
        '''构建一个词汇表 vocab，其中每个元素表示一个词，以及其对应的词的结尾标记。'''
        self.byte_encoder = bytes_to_unicode()#文本转unicode
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}#反转的字典
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')#这行代码打开 BPE 文件（使用 gzip 模块），读取文件内容并解码为 UTF-8 格式的字符串。然后，使用 split('\n') 方法将字符串按换行符拆分为一个字符串列表 merges。
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]#遍历每个元素，切分，将结果转换为元组，终得到的结果是一个由拆分后的元组构成的新列表。
        vocab = list(bytes_to_unicode().values())#返回的字节编码到 Unicode 字符的映射字典中的所有值提取出来，并将其转换为一个列表 vocab
        vocab = vocab + [v+'</w>' for v in vocab]#将 vocab 列表中的每个元素（Unicode 字符）追加一个 '</w>' 后缀，表示这是一个词的结尾标记。
        for merge in merges:
            vocab.append(''.join(merge))#将 BPE 合并列表 merges 中的每个合并元组连接起来，并将连接后的字符串作为一个词添加到词汇表 vocab 中。
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])#起始和结束标记
        # 将词汇表中的每个词与对应的索引建立映射关系，形成encoder字典
        self.encoder = dict(zip(vocab, range(len(vocab))))

        # 利用字典推导式，将encoder字典的键值对颠倒，形成decoder字典
        self.decoder = {v: k for k, v in self.encoder.items()}

        # 将merges列表的每个元素与对应的索引建立映射关系，形成bpe_ranks字典
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}#初始化cache字典，将特殊标记字符串与其本身作为键值对存储
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)
        #'s,'t,'re,'ve,'m,'ll,'d是缩写形式的词，如"is"、"are"、"have"等。 [\p{L}]+匹配一个或多个Unicode字母。 [\p{N}]匹配任意一个Unicode数字。 [^\s\p{L}\p{N}]+匹配不包含空格、字母和数字的字符。
        #通过使用re.compile()函数，并传递正则表达式模式的字符串和标志参数re.IGNORECASE（表示忽略大小写），我们将创建一个可以在文本中查找和匹配这些模式的正则表达式对象。
    def bpe(self, token):
        '''为了处理一种特殊情况，即当token不完整时，例如"hello"被划分为"hell"和"o"，我们需要将其合并为一个完整的单词。'''
        if token in self.cache:#为了处理一种特殊情况，即当token不完整时，例如"hello"被划分为"hell"和"o"，我们需要将其合并为一个完整的单词。
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)
        #如果pairs为空，意味着word只有一个字符，也就是token本身。在这种情况下，方法返回token加上'</w>'，表示单词结束。
        #这样做是为了处理特殊情况，即当输入的token只有一个字符时，无需再进一步处理和划分，直接加上'</w>'表示单词结束即可。
        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))#找到pairs中出现频率最低的字符对bigram。这里使用了一个lambda函数作为key参数，以获取每个字符对对应的频率值。
            if bigram not in self.bpe_ranks:#如果找不到对应的频率值，就退出循环。
                break
            first, second = bigram#分解为两个字符
            #创建一个空的new_word列表，并设置索引i的初始值为0。接下来，用一个while循环遍历word中的字符。
            #如果在遍历过程中遇到了first字符，且first和second字符相邻，就将它们合并成一个新的字符，并将索引i增加2。
            #否则，将当前字符word[i]添加到new_word中，并将索引i增加1。
            #完成一轮遍历后，将new_word转换为元组并赋值给word，以准备进行下一轮划分。
            #如果划分后的word长度为1，即只剩下一个字符，说明划分过程结束，退出循环。
            #否则，更新pairs，继续下一轮划分。
            new_word = []
            i = 0
            while i < len(word):#
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()#数据清洗后转换为小写
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        #将text转换为一个字节数组，并使用self.byte_decoder字典将每个字符解码为原始的UTF-8字节。
        #然后，使用decode()函数将字节数组转换为字符串，并指定使用UTF-8编码。errors="replace"参数表示在解码时，如果遇到无法解码的字节，则将其替换为特定的替换字符。
        #最后，通过调用replace()函数，将BPE标记'</w>'替换为空格
        return text
