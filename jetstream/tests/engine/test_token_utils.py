import os
import unittest
from typing import List

from sentencepiece import SentencePieceProcessor
from jetstream.engine import tokenizer_pb2, token_utils


class SPTokenizer:
   """Tokenier used in original llama2 git"""

   def __init__(self, tokenizer_path: str):
       self.tokenizer = SentencePieceProcessor(model_file=tokenizer_path)
       assert self.tokenizer.vocab_size() == self.tokenizer.get_piece_size()

   def decode(self, t: List[int]) -> str:
       token = self.tokenizer.decode(t)
       return token


class JetStreamTokenizer:
   """Tokenier used in JetStream before mix_token"""


   def __init__(self, tokenizer_path: str):
    metadata = tokenizer_pb2.TokenizerParameters(path=tokenizer_path)
    self.vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)

   def decode(self, t: int) -> str:
    token = self.vocab.tokenizer.IdToPiece(t)
    token = token.replace('▁', ' ')
    return token      


class TokenUtilsTest(unittest.TestCase):
    def setup(self):
        tokenizer_path = "tokenizer.model"
        current_dir = os.path.dirname(__file__)
        tokenizer_path = os.path.join(current_dir, tokenizer_path)
        print(f"model_path: {tokenizer_path}")
        assert os.path.isfile(tokenizer_path), f"file not found tokenizer_path: {tokenizer_path}"
        self.sp_tokenizer = SPTokenizer(tokenizer_path)
        self.jt_tokenizer = JetStreamTokenizer(tokenizer_path)

    def test_decode_vs_piece(self):
       self.setup()
       tokens = [304, 13, 2266, 526, 777, 9590, 2020, 29901]
       expeted_sp_output = []
       jt_output = []
       print(f"jt_output: {jt_output}")
       for t in tokens:
           expeted_sp_output.append(self.sp_tokenizer.decode([t]))
           jt_output.append(self.jt_tokenizer.decode(t))

       self.assertNotEqual(jt_output, expeted_sp_output)   


    def test_mix_decode(self):
       self.setup()
       for n in range(0, self.sp_tokenizer.tokenizer.vocab_size()):
          # From decode function
          decode_output = self.sp_tokenizer.decode([n])
          # From IdToPiece function
          piece_output = self.jt_tokenizer.decode(n)
          # Mix output from decode and IdToPiece 
          mix_output = token_utils.mix_decode(vocab = self.jt_tokenizer.vocab, tok_id = n)
          if piece_output.lstrip() == decode_output:
            self.assertEqual(mix_output, piece_output)
          else:
            self.assertEqual(mix_output, decode_output)    


    def test_sp_vs_seqio(self):
       self.setup()
       for n in range(0, self.sp_tokenizer.tokenizer.vocab_size()):
          sp_t = self.sp_tokenizer.decode([n])
          seqio_t = self.jt_tokenizer.vocab.tokenizer.decode([n])
          self.assertEqual(sp_t, seqio_t)

    def test_underscore_in_output(self):
       self.setup()
       n = 21326
       mix_output = token_utils.mix_decode(vocab = self.jt_tokenizer.vocab, tok_id = n)
       decode_output = self.sp_tokenizer.decode([n])  
       self.assertEqual(mix_output, " `__")  
       self.assertEqual(mix_output.lstrip(), decode_output)


if __name__ == '__main__':
   unittest.main()
 