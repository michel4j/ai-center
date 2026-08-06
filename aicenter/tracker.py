import numpy as np


class ArbitraryTranslatingObject:

    def __init__(self, track_id, last_cx, last_cy, radius, angular_offset):
        self.track_id = track_id
        self.last_cx = last_cx  # Last known absolute X center
        self.last_cy = last_cy  # Last known absolute Y center
        self.radius = radius  # Distance from fixed axis center
        self.angular_offset = angular_offset  # Angular offset relative to system wheel
        self.lost_frames = 0


class AdvancedConstrainedTracker:

    def __init__(
        self,
        fixed_center: tuple[int, int] = (960, 540),
        r_threshold: int = 30,
        angle_threshold: float = 15,
        max_lost: int = 60,
    ):
        """
        :param fixed_center: (X, Y) pixel coordinates of the stable rotation axis
        :param r_threshold: Maximum allowed radial distance difference for matching
        :param angle_threshold: Maximum allowed angular offset difference for matching
        :param max_lost: Maximum number of frames an object can be lost before its track is deleted

        """
        self.tracks = []
        self.next_id = 0
        self.center_x, self.center_y = fixed_center
        self.r_threshold = r_threshold
        self.angle_threshold = angle_threshold
        self.max_lost = max_lost
        self.last_global_angle = None

    def update(self, objects: list[dict], current_global_angle: float):
        """
        Update tracking state and object IDs
        :param objects: list of dicts, each containing at least 'box': [x1, y1, x2, y2]
        :param current_global_angle: float, system rotation angle in
        degrees Returns: list of dicts matching the input structure, with a
        unique 'id' key injected
        """

        # Detect Phase State
        is_rotating = True
        if self.last_global_angle is not None:
            if abs(current_global_angle - self.last_global_angle) < 0.01:
                is_rotating = False

        self.last_global_angle = current_global_angle

        # Extract current frame polar and Cartesian states
        detections = []
        for obj in objects:
            box = obj["box"]
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0

            dx = cx - self.center_x
            dy = cy - self.center_y

            radius = np.sqrt(dx**2 + dy**2)
            detected_angle = np.degrees(np.arctan2(dy, dx)) % 360
            ang_offset = (detected_angle - current_global_angle) % 360

            detections.append({
                "original_obj": obj,
                "cx": cx,
                "cy": cy,
                "r": radius,
                "offset": ang_offset,
                "matched": False,
            })

        tracked_output_objects = []
        dx_translations = []
        dy_translations = []

        # Increment age of all tracks
        for track in self.tracks:
            track.lost_frames += 1

        # Primary Matching Loop
        for track in self.tracks:
            best_match_idx = None
            min_cost = float("inf")

            for idx, det in enumerate(detections):
                if det["matched"]:
                    continue

                # Check angular phase alignment
                angle_diff = min(
                    abs(track.angular_offset - det["offset"]),
                    360 - abs(track.angular_offset - det["offset"]),
                )

                r_diff = abs(track.radius - det["r"])

                # Enforce rigid structural limits strictly during active rotation phases
                r_gating = r_diff < self.r_threshold if is_rotating else True

                if angle_diff < self.angle_threshold and r_gating:
                    cost = angle_diff + (r_diff * 0.5)
                    if cost < min_cost:
                        min_cost = cost
                        best_match_idx = idx

            # Process Match & Calculate Linear Vectors
            if best_match_idx is not None:
                det = detections[best_match_idx]
                det["matched"] = True
                track.lost_frames = 0

                # If paused, document arbitrary 2D translation vectors
                if not is_rotating:
                    dx_translations.append(det["cx"] - track.last_cx)
                    dy_translations.append(det["cy"] - track.last_cy)

                # Save state changes
                track.last_cx = det["cx"]
                track.last_cy = det["cy"]
                track.radius = det["r"]
                # Update angular offset with a weighted average to smooth out jitter
                # The weight (0.2) can be tuned for responsiveness vs. stability
                track.angular_offset = (0.8 * track.angular_offset + 0.2 * det["offset"]) % 360

                # Format output dictionary using shallow copy to protect source keys
                matched_obj = det["original_obj"].copy()
                matched_obj["id"] = track.track_id
                tracked_output_objects.append(matched_obj)

        # Dead Reckoning Updates for Occluded Objects
        if not is_rotating and len(dx_translations) > 0:
            # Calculate the global 2D median shift vector across the scene
            global_dx = np.median(dx_translations)
            global_dy = np.median(dy_translations)

            for track in self.tracks:
                if track.lost_frames > 0:
                    # Update its absolute Cartesian projection position
                    track.last_cx += global_dx
                    track.last_cy += global_dy

                    # Compute projected Polar coordinates relative to the stable axis point
                    new_dx = track.last_cx - self.center_x
                    new_dy = track.last_cy - self.center_y

                    track.radius = np.sqrt(new_dx**2 + new_dy**2)
                    proj_angle = np.degrees(np.arctan2(new_dy, new_dx)) % 360
                    track.angular_offset = (
                        proj_angle - current_global_angle
                    ) % 360

        # Re-register emerging or newly detected items
        for det in detections:
            if not det["matched"]:
                is_old_track = False
                for track in self.tracks:
                    if track.lost_frames > 0:
                        angle_diff = min(
                            abs(track.angular_offset - det["offset"]),
                            360 - abs(track.angular_offset - det["offset"]),
                        )
                        r_diff = abs(track.radius - det["r"])

                        if (
                            angle_diff < self.angle_threshold
                            and r_diff < self.r_threshold
                        ):
                            track.lost_frames = 0
                            track.last_cx = det["cx"]
                            track.last_cy = det["cy"]

                            matched_obj = det["original_obj"].copy()
                            matched_obj["id"] = track.track_id
                            tracked_output_objects.append(matched_obj)

                            is_old_track = True
                            break

                if not is_old_track:
                    new_track = ArbitraryTranslatingObject(
                        self.next_id,
                        det["cx"],
                        det["cy"],
                        det["r"],
                        det["offset"],
                    )
                    self.tracks.append(new_track)

                    new_obj = det["original_obj"].copy()
                    new_obj["id"] = self.next_id
                    tracked_output_objects.append(new_obj)

                    self.next_id += 1

        # Clear dead states
        self.tracks = [t for t in self.tracks if t.lost_frames <= self.max_lost]
        return tracked_output_objects
