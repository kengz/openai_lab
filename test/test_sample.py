import unittest


class SampleTest(unittest.TestCase):

  def testSample(self):
    foo = 1
    self.assertEqual(foo, 1)
