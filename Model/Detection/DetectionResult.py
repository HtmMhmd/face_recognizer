class DetectionResult:
    def __init__(self, boxes=None, scores=None, class_ids=None):
        self.boxes = boxes if boxes is not None else []
        self.scores = scores if scores is not None else []
        self.class_ids = class_ids if class_ids is not None else []
        self.names = 'Face'

    def __getitem__(self, index):
        if isinstance(index, int):
            return DetectionResult(
                boxes=[self.boxes[index]],
                scores=[self.scores[index]],
                class_ids=[self.class_ids[index]]
            )
        elif isinstance(index, slice):
            return DetectionResult(
                boxes=self.boxes[index],
                scores=self.scores[index],
                class_ids=self.class_ids[index]
            )
        else:
            raise TypeError("Index must be an integer or a slice")

    def __len__(self):
        return len(self.boxes)

    def __repr__(self):
        return f"DetectionResult(boxes={self.boxes}, scores={self.scores}, class_ids={self.class_ids})"

    def add(self, box, score, class_id):
        self.boxes.append(box)
        self.scores.append(score)
        self.class_ids.append(class_id)