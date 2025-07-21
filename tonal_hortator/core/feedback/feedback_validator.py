from typing import Any, List, Optional


class FeedbackValidator:
    VALID_QUERY_TYPES = ["artist_specific", "similarity", "general"]
    VALID_ACTIONS = ["like", "dislike", "skip", "block", "favorite"]
    VALID_FEEDBACK_TYPES = ["like", "dislike", "block"]
    VALID_PREFERENCE_TYPES = ["string", "integer", "float", "boolean", "json"]

    @staticmethod
    def validate_query_type(query_type: str) -> bool:
        return query_type in FeedbackValidator.VALID_QUERY_TYPES

    @staticmethod
    def validate_user_rating(user_rating: Optional[int]) -> bool:
        return user_rating is None or (0 <= user_rating <= 5)

    @staticmethod
    def validate_user_actions(user_actions: Optional[List[str]]) -> bool:
        if user_actions is None:
            return True
        return all(action in FeedbackValidator.VALID_ACTIONS for action in user_actions)

    @staticmethod
    def validate_playlist_length(length: Optional[int]) -> bool:
        return length is None or length >= 0

    @staticmethod
    def validate_similarity_threshold(threshold: Optional[float]) -> bool:
        return threshold is None or (0.0 <= threshold <= 1.0)

    @staticmethod
    def validate_search_breadth(breadth: Optional[int]) -> bool:
        return breadth is None or breadth >= 1

    @staticmethod
    def validate_feedback_type(feedback: str) -> bool:
        return feedback in FeedbackValidator.VALID_FEEDBACK_TYPES

    @staticmethod
    def validate_track_id(track_id: str) -> bool:
        return track_id.strip() != ""

    @staticmethod
    def validate_preference_type(preference_type: str) -> bool:
        return preference_type in FeedbackValidator.VALID_PREFERENCE_TYPES

    @staticmethod
    def validate_preference_value(value: Any, preference_type: str) -> bool:
        if preference_type == "integer":
            return isinstance(value, int)
        elif preference_type == "float":
            return isinstance(value, (int, float))
        elif preference_type == "boolean":
            return isinstance(value, bool)
        elif preference_type == "string":
            return isinstance(value, str)
        elif preference_type == "json":
            # JSON values can be any serializable type
            return True
        return False
