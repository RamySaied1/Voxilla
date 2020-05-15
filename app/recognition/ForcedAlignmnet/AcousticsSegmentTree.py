class Segment():
    def __init__(self,label,_range):
        self.label = label
        self.range = _range

    def __hash__(self):
        return hash(self.label)

class AcousticsSegmentTree():
    def __init__(self):
        self._clean()

    def _clean(self):
        self.levels = []
        self.levelsNames = []
        self.levelsIndx = []
        self.groupings = []

    def build(self,groupings,levelsNames=None):
        self._clean()
        self.groupings = groupings
        self.levelsNames = levelsNames if levelsNames is not None else map(str,range(len(groupings)))
        self.levelsIndx = dict(zip(levelsNames,range(len(groupings))))
        self.levelsIndx[levelsNames[-1]] = self.levelsIndx[-1] = len(groupings)-1

        for grouping in groupings:
            i = 0 
            level = []
            for count,label in grouping:
                level.append(Segment(label,(i,i+count)))
                i += count 
            self.levels.append(level)

            # print([seg.range for seg in level],len(level))
    
    def get_level_ranges(self,levelBegin,levelEnd):
        return [ self.get_seg_range(i,levelBegin,levelEnd) for i in range(len(self.levels[self.levelsIndx[levelBegin]])) ]
    
    def get_level_labels(self,level):
        return [ seg.label for seg in self.levels[self.levelsIndx[level]] ]

    def get_seg_range(self,elemI,levelBegin,levelEnd):
        levelBegin = self.levelsIndx[levelBegin]
        levelEnd = self.levelsIndx[levelEnd]
        assert(levelBegin < levelEnd)
        leftI = elemI
        rightI = elemI+1
        for li in range(levelBegin,levelEnd+1):
            leftI = self.levels[li][leftI].range[0]
            rightI = self.levels[li][rightI-1].range[1]
        return (leftI,rightI)
