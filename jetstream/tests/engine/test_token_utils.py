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
   """Tokenier used in JetStream"""


   def __init__(self, tokenizer_path: str):
       metadata = tokenizer_pb2.TokenizerParameters(path=tokenizer_path)
       self.vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)

   def decode(self, t: List[int]) -> str:
       token = self.vocab.tokenizer.IdToPiece(t)
       token = token.replace('▁', ' ').replace('_', ' ')
       return token      
   

class TokenUtilsTest(unittest.TestCase):
    def setup(self):
       # Please replace with your own tokenizer data set
       tokenizer_path = "tokenizer.model"
       self.sp_tokenizer = SPTokenizer(tokenizer_path)
       self.jt_tokenizer = JetStreamTokenizer(tokenizer_path)


    def test_decode(self):
       self.setup()
       # Longer token ids for prompt: "I believe the meaning of life is"
       #tokens = [304, 1284, 6437, 29892, 22722, 29892, 322, 6095, 5589, 358, 29889, 2266, 526, 777, 9590, 2020, 29901, 13, 13, 29896, 29889, 15247, 4220, 29901, 15950, 263, 4060, 310, 6437, 4076, 2834, 6593, 322, 5305, 29889, 739, 6911, 15724, 731, 14433, 322, 664, 7113, 3657, 15387, 963, 29892, 607, 508, 3275, 304, 263, 4060, 310, 6095, 5589, 358, 322, 26470, 29889, 13, 29906, 29889, 379, 932, 3335, 29901, 379, 932, 3335, 338, 263, 15281, 5199, 817, 29892, 322, 372, 338, 18853, 363, 12463, 1532, 29899, 915, 292, 29889, 349, 1295, 26420, 14188, 322, 27482, 393, 6963, 15331, 322, 22722, 508, 26371, 749, 2834, 26470, 322, 12463, 11029, 310, 2834, 29889, 13, 29941, 29889, 23004, 5589, 358, 29901, 23004, 5589, 358, 338, 278, 11223, 310, 12709, 358, 393, 5304, 515, 3657, 15387, 697, 29915, 29879, 14433, 322, 12359, 26420, 697, 29915, 29879, 1209, 1080, 29889, 739, 338, 278, 4060, 310, 26470, 322, 2793, 358, 393, 5304, 515, 8471, 263, 2834, 393, 338, 1565, 304, 6743, 761, 29889, 13, 29946, 29889, 16224, 14321, 29901, 16224, 14321, 322, 1583, 29899, 326, 16123, 882, 526, 18853, 363, 263, 6095, 5589, 292, 2834, 29889, 29257, 716, 25078, 29892, 14338, 716, 2299, 1169, 29892, 322, 975, 11506, 18066, 267, 508, 1371, 15724, 6548, 322, 2693, 408, 2305, 29889, 13, 29945, 29889, 6376, 800, 14587, 29901, 3767, 549, 21702, 411, 3942, 29892, 7875, 29892, 322, 18012, 6743, 526, 12187, 363, 263, 9796, 322, 6095, 5589, 292, 2834, 29889, 10307, 12368, 3867, 23023, 1848, 2304, 29892, 18708, 3527, 29892, 322, 263, 4060, 310, 23329, 29889, 13, 29953, 29889, 2866, 3224, 29901, 341, 5086, 263, 6374, 10879, 373, 278, 3186, 322, 17737, 17068, 304, 1554, 7200, 1135, 6743, 761, 508, 2367, 2834, 6593, 322, 6437, 29889, 402, 4357, 1250, 304, 278, 7881, 29892, 27886, 3241, 29892, 470, 12359, 26420, 263, 6413, 393, 23633, 12459, 508, 3867, 263, 4060, 310, 6095, 5589, 358, 322, 6437, 29889, 13, 29955, 29889, 28224, 5597, 29901, 28224, 819, 3277, 716, 2712, 29892, 3902, 8253, 716, 7600, 29892, 322, 1811, 716, 14188, 508, 788, 8261, 2264, 322, 10809, 304, 2834, 29889, 3201, 955, 292, 29892, 6509, 716, 4185, 1973, 29892, 322, 3033, 6751, 297, 716, 298, 20838, 583, 508, 2545, 4858, 4029, 466, 787, 322, 1653, 1833, 292, 2626, 3842, 29889, 13, 29947, 29889, 1632, 271, 4279, 29901, 29124, 18499, 20715, 4279, 322, 5108, 362, 363, 278, 1781, 2712, 297, 2834, 508, 1371, 15724, 18834, 403, 263, 6374, 3458, 842, 322, 11188, 278, 15409, 310, 2834, 29889, 383, 542, 4746, 373, 278, 2198, 3256, 322, 4653, 292, 20715, 4279, 363, 825, 697, 756, 508, 3275, 304, 7621, 22722, 322, 6095, 5589, 358, 29889, 13, 29929, 29889, 20152, 1319, 2264, 29901, 28265, 2198, 322, 3458, 1319, 297, 278, 3256, 508, 1371, 15724, 11188, 278, 15409, 310, 2834, 322, 1284, 15331, 297, 1432, 3250, 27482, 29889, 29124, 18499, 3458, 1319, 2264, 508, 1371, 10032, 22884, 322, 7910, 12463, 1532, 29899, 915, 292, 29889, 13, 29896, 29900, 29889, 5682, 4135, 29901, 951, 5555, 263, 1833, 292, 25000, 393, 9432, 29879, 697, 29915, 29879, 1819, 322, 1209, 1080, 508, 2367, 2834, 6593, 322, 6437, 8724, 697, 29915, 29879, 1914, 11747, 267, 8357, 29889, 26221, 1554, 393, 674, 714, 4230, 6743, 761, 508, 3867, 263, 4060, 310, 6095, 5589, 358, 322, 6437, 29889, 13, 13, 797, 15997, 29892, 9138, 6437, 29892, 22722, 29892, 322, 6095, 5589, 358, 297, 2834, 338, 263, 7333, 322, 373, 17696, 16342, 29889, 739, 6858, 1583, 29899, 1450, 8326, 404, 29892, 1583, 29899, 999, 1464, 29892, 322, 263, 17762, 2264, 304, 3133, 5794, 6548, 322, 7744, 29889, 2648, 12359, 26420, 14188, 322, 27482, 393, 6963, 15331, 322, 6095, 5589, 358, 29892, 15724, 508, 1653, 263, 6593, 1319, 322, 6095, 5589, 292, 2834, 393, 9432, 29879, 1009, 1819, 322, 1209, 1080, 29889]
       tokens = [304, 13, 2266, 526, 777, 9590, 2020, 29901]
       expeted_sp_output = []
       jt_output = []
       print(f"jt_output: {jt_output}")
       for t in tokens:
           expeted_sp_output.append(self.sp_tokenizer.decode(t))
           jt_output.append(self.jt_tokenizer.decode(t))

       self.assertEqual(jt_output, expeted_sp_output)   

          
if __name__ == '__main__':
   unittest.main()