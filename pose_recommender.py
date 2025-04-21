import random
from typing import List

class PoseRecommender:
    def __init__(self, mode: str):
        self.mode = mode
        self._init_mode()
        self.current_index = 0  # For Class Mode

    def _init_mode(self):
        if self.mode == "Standing-Only":
            self.full_pool = ["Chair Pose", "Lord of the Dance Pose", "Tree Pose", 
                              "Warrior 1", "Warrior 2", "Warrior 3"]
            self.remaining_poses = self.full_pool.copy()

        elif self.mode == "Hard":
            self.full_pool = ["Boat Pose", "Chair Pose", "Lord of the Dance Pose", 
                              "Side Plank Pose", "Tree Pose", "Warrior 3"]
            self.remaining_poses = self.full_pool.copy()

        elif self.mode == "Class":
            self.sequence = ["Fish Pose", "Boat Pose", "Side Plank Pose", "Child Pose", 
                             "Downward Facing Dog", "Warrior 1", "Warrior 2", "Chair Pose", 
                             "Tree Pose", "Warrior 3", "Lord of the Dance Pose", "Sitting Pose"]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_next_pose(self, curr_pose: str) -> str:
        if self.mode in ["Standing-Only", "Hard"]:
            return self._get_next_random_pose(curr_pose)
        elif self.mode == "Class":
            return self._get_next_class_pose()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _get_next_random_pose(self, curr_pose: str) -> str:
        if curr_pose in self.remaining_poses:
            self.remaining_poses.remove(curr_pose)

        if not self.remaining_poses:
            self.remaining_poses = self.full_pool.copy()
            self.remaining_poses.remove(curr_pose)

        return random.choice(self.remaining_poses)

    def _get_next_class_pose(self) -> str:
        pose = self.sequence[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.sequence)
        return pose