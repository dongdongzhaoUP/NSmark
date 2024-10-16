import hashlib
import hmac
import sys

from torch import range


def Sign(m, k):
    def generate_signature(message, secret_key):
        data_bytes = message.encode("utf-8")
        signature = hmac.new(
            secret_key.encode("utf-8"), data_bytes, hashlib.sha256
        ).hexdigest()
        return signature

    signature = generate_signature(m, k)

    def Str_encode(s: str, length: int, rule="utf-8"):
        sc = s.encode(rule)
        bc = [bin(int(i))[2:].rjust(8, "0") for i in sc]
        bc = "".join(bc)
        bc = [(1 if i == "1" else -1) for i in bc]
        return bc[:length]

    sig = Str_encode(signature, 256)
    return sig


def Trigger_gen(sig, len_vocab):
    idx = hash(str(sig)) % ((sys.maxsize + 1) * 2) % len_vocab
    return idx


def Select_algo(sig, q, len_ds):
    index_list = []
    h = hash(str(sig)) % ((sys.maxsize + 1) * 2)
    for i in range(1, q):
        h = hash(str(h)) % ((sys.maxsize + 1) * 2)
        idx = h % len_ds
        index_list.append(idx)
    return index_list


# 要签名的数据
message = "Hello, World!"
# 私钥
secret_key = "my_secret_key"

sig = Sign(message, secret_key)
print(sig)
trigger_idx = Trigger_gen(sig, 30000)
# print(trigger_idx)
D_V = Select_algo(sig, 1500, 60000)
print(D_V)
print(len(D_V))
print(len(list(set(D_V))))
