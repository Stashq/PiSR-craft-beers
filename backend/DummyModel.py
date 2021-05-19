class DummyModel:
    def __init__(self):
        pass

    def predict(self, user_id):
        dummy_data = {
            10: {
                "ForYou": [1, 2, 3, 4],
                "popular": [5, 6, 7, 8],
                "bestRating": [321, 22, 23, 233],
            },
            11: {
                "ForYou": [412, 321, 1233, 23],
                "popular": [2, 2, 2, 2],
                "bestRating": [3, 3, 3, 3],
            },
            12: {
                "ForYou": [1, 1, 1, 1],
                "popular": [2, 2, 2, 2],
                "bestRating": [3, 3, 3, 3],
            },
        }
        return dummy_data[user_id]
