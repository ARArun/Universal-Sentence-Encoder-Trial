#! ../ten/bin/python3

import hashlib
handle = "https://tfhub.dev/google/universal-sentence-encoder-large/4"
print(hashlib.sha1(handle.encode("utf8")).hexdigest())