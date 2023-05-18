# This file is a helper script for the evaluation of the results

import distance
from pyymatcher import PyyMatcher
from skimage.filters import threshold_otsu
import numpy as np

s1 = 'string1'
s2 = 'string2'

s1 = s1.lower()
s2 = s2.lower()


levenshteinDist = distance.levenshtein(s1, s2)
levenshteinSim = 1 - levenshteinDist/max(len(s1), len(s2))
gestaltObj = PyyMatcher(s1, s2)
gestalt = gestaltObj.ratio()


print ('Levenshtein: ' + str(levenshteinSim))
print ('Gestalt: ' + str(gestalt))


textSimilarities = [0.1, 0.2, 0.3, 0.4]
textSimilarities = np.array(textSimilarities)

threshold = threshold_otsu(textSimilarities)
print ('Threshold: ' + str(threshold))