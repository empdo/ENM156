from status_mapping import interpret_prediction


def test_known_classes():
    expected = {"felm4":"fault1", "korrm4":"normal", "medm8":"normal",
                "utanm8":"fault2", "bakgrund":"idle", "background":"idle"}
    for label, status in expected.items():
        assert interpret_prediction(label)["status"] == status
    assert interpret_prediction("ny_okand_klass")["status"] == "uncertain"
    print("Alla statusmappningar OK")

if __name__ == "__main__":
    test_known_classes()
