class DetectionResult:
    def __init__(self, boxes=None, scores=None, class_ids=None):
        self.boxes = boxes if boxes is not None else []
        self.scores = scores if scores is not None else []
        self.class_ids = class_ids if class_ids is not None else []
        self.names = 'Face'
        self.n_faces = 0

    def __getitem__(self, key):
        if isinstance(key, int):
            return DetectionResult(
                boxes=[self.boxes[key]],
                scores=[self.scores[key]],
                class_ids=[self.class_ids[key]]
            )
        elif isinstance(key, slice):
            return DetectionResult(
                boxes=self.boxes[key],
                scores=self.scores[key],
                class_ids=self.class_ids[key]
            )
        elif isinstance(key, str):
            if key == "boxes" or key == "bbox":
                return self.boxes
            elif key == "scores":
                return self.scores
            elif key == "class_ids":
                return self.class_ids
            else:
                raise KeyError(f"Invalid key: {key}")
        else:
            raise TypeError("Index must be an integer, slice, or string")

    def __len__(self):
        return len(self.boxes)

    def __repr__(self):
        return f"DetectionResult(boxes={self.boxes}, scores={self.scores}, class_ids={self.class_ids})"

    def add(self, box, score, class_id):
        if (box is not None )and (score is not None) and (class_id is not None):    
            self.n_faces += 1
            self.boxes.append(box)
            self.scores.append(score)
            self.class_ids.append(class_id)
        else:
            raise ValueError("box, score and class_id must not be None")
    
    def reset(self):
        self.boxes = []
        self.scores = []
        self.class_ids = []
        self.n_faces = 0