import unittest
import joblib
import spam

class SpamTests(unittest.TestCase):
    def test_duplicates(self):
        self.assertEqual(len(spam.deduplicate([(0,'Hello  world'),(0,'HELLO world')])),1)
    def test_conflicting_labels(self):
        with self.assertRaises(ValueError): spam.deduplicate([(0,'Hello'),(1,'hello')])
    def test_invalid_input(self):
        for value in [None, 3, '', '  ', 'x'*10001]:
            with self.assertRaises(ValueError): spam.predict(value)
    def test_persistence(self):
        bundle = joblib.load(spam.ROOT/'artifacts/model.joblib')
        self.assertEqual(spam.predict('Meeting at noon',bundle),spam.predict('Meeting at noon'))
    def test_no_vocabulary_leakage(self):
        model = spam.make_pipeline(spam.TfidfVectorizer(),spam.LogisticRegression())
        model.fit(['legitimate meeting','claim prize','meeting today','win prize'],[0,1,0,1])
        model.predict(['unseentesttoken'])
        self.assertNotIn('unseentesttoken',model[0].vocabulary_)

if __name__ == '__main__': unittest.main()
