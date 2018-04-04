def make_human(i):
    import random

    def getRandomValue(minValue, maxValue, middleValue, sigmaFactor = 0.2):
        import random
        rangeWidth = float(abs(maxValue - minValue))
        sigma = sigmaFactor * rangeWidth
        randomVal = random.gauss(middleValue, sigma)
        if randomVal < minValue:
            randomVal = minValue + abs(randomVal - minValue)
        elif randomVal > maxValue:
            randomVal = maxValue - abs(randomVal - maxValue)
        return max(minValue, min(randomVal, maxValue))

    symmetry = 0.75

    modifierGroups = []
    modifierGroups = modifierGroups + ['macrodetails', 'macrodetails-universal', 'macrodetails-proportions']
    modifierGroups = modifierGroups + ['macrodetails-height']
    modifierGroups = modifierGroups + ['eyebrows', 'eyes', 'chin',
                     'forehead', 'head', 'mouth', 'nose', 'neck', 'ears',
                     'cheek']
    modifierGroups = modifierGroups + ['pelvis', 'hip', 'armslegs', 'stomach', 'breast', 'buttocks', 'torso']

    modifiers = []
    for mGroup in modifierGroups:
        modifiers = modifiers + MHScript.human.getModifiersByGroup(mGroup)
    # Make sure not all modifiers are always set in the same order
    # (makes it easy to vary dependent modifiers like ethnics)
    random.shuffle(modifiers)

    randomValues = {}
    for m in modifiers:
        if m.fullName not in randomValues:
            randomValue = None
            if m.groupName == 'head':
                sigma = 0.1
            elif m.fullName in ["forehead/forehead-nubian-less|more", "forehead/forehead-scale-vert-less|more"]:
                sigma = 0.02
                # TODO add further restrictions on gender-dependent targets like pregnant and breast
            elif "trans-horiz" in m.fullName or m.fullName == "hip/hip-trans-in|out":
                if symmetry == 1:
                    randomValue = m.getDefaultValue()
                else:
                    mMin = m.getMin()
                    mMax = m.getMax()
                    w = float(abs(mMax - mMin) * (1 - symmetry))
                    mMin = max(mMin, m.getDefaultValue() - w/2)
                    mMax = min(mMax, m.getDefaultValue() + w/2)
                    randomValue = getRandomValue(mMin, mMax, m.getDefaultValue(), 0.1)
            elif m.groupName in ["forehead", "eyebrows", "neck", "eyes", "nose", "ears", "chin", "cheek", "mouth"]:
                sigma = 0.1
            elif m.groupName == 'macrodetails':
                # TODO perhaps assign uniform random values to macro modifiers?
                #randomValue = random.random()
                sigma = 0.3
            #elif m.groupName == "armslegs":
            #    sigma = 0.1
            else:
                #sigma = 0.2
                sigma = 0.1

            if randomValue is None:
                randomValue = getRandomValue(m.getMin(), m.getMax(), m.getDefaultValue(), sigma)   # TODO also allow it to continue from current value?
            randomValues[m.fullName] = randomValue
            symm = m.getSymmetricOpposite()
            if symm and symm not in randomValues:
                if symmetry == 1:
                    randomValues[symm] = randomValue
                else:
                    m2 = MHScript.human.getModifier(symm)
                    symmDeviation = float((1-symmetry) * abs(m2.getMax() - m2.getMin()))/2
                    symMin =  max(m2.getMin(), min(randomValue - (symmDeviation), m2.getMax()))
                    symMax =  max(m2.getMin(), min(randomValue + (symmDeviation), m2.getMax()))
                    randomValues[symm] = getRandomValue(symMin, symMax, randomValue, sigma)

    if randomValues.get("macrodetails/Gender", 0) > 0.5 or \
       randomValues.get("macrodetails/Age", 0.5) < 0.2 or \
       randomValues.get("macrodetails/Age", 0.7) < 0.75:
        # No pregnancy for male, too young or too old subjects
        if "stomach/stomach-pregnant-decr|incr" in randomValues:
            randomValues["stomach/stomach-pregnant-decr|incr"] = 0

    _tmp = MHScript.human.symmetryModeEnabled
    MHScript.human.symmetryModeEnabled = False
    for mName, val in randomValues.items():
       try:
          MHScript.human.getModifier(mName).setValue(val)
       except:
          pass

    import numpy as np
    save_folder = '/Users/kostrikov/tmp/'
    MHScript.human.applyAllTargets()
    MHScript.human.symmetryModeEnabled = _tmp
    MHScript.saveObj('object{}'.format(i), save_folder)
    np.save(save_folder + 'labels{}'.format(i), randomValues)

for i in range(10):
   make_human(i)
