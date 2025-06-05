#centroid distance with the closest player distance
import numpy as np
import supervision as sv

def resolve_goalkeepers_team_id(
    players: sv.Detections,
    goalkeepers: sv.Detections
) -> np.ndarray:
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    team_0_xy = players_xy[players.class_id == 0]
    team_1_xy = players_xy[players.class_id == 1]

    team_0_centroid = team_0_xy.mean(axis=0)
    team_1_centroid = team_1_xy.mean(axis=0)

    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        # Distance to centroids
        dist_centroid_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_centroid_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)

        # Distance to nearest player from each team
        nearest_0 = np.min(np.linalg.norm(team_0_xy - goalkeeper_xy, axis=1))
        nearest_1 = np.min(np.linalg.norm(team_1_xy - goalkeeper_xy, axis=1))

        # Weighted score: closer centroid + closer nearest player
        score_0 = dist_centroid_0 * 0.5 + nearest_0 * 0.5
        score_1 = dist_centroid_1 * 0.5 + nearest_1 * 0.5

        goalkeepers_team_id.append(0 if score_0 < score_1 else 1)

    return np.array(goalkeepers_team_id)
