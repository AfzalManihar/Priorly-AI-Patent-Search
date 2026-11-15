def test_pipeline_settings_expose_only_gpt_oss_120b(client):
    response = client.get("/api/settings")
    assert response.status_code == 200
    assert response.json()["llm_model"] == "openai/gpt-oss-120b"


def test_pipeline_settings_accept_only_gpt_oss_120b(client):
    accepted = client.put(
        "/api/settings",
        json={"llm_model": "openai/gpt-oss-120b"},
    )
    assert accepted.status_code == 200
    assert accepted.json()["llm_model"] == "openai/gpt-oss-120b"

    assert client.put(
        "/api/settings",
        json={"llm_model": "Groq, llama-3.3-70b"},
    ).status_code == 422

    assert client.put(
        "/api/settings",
        json={"llm_model": "OpenAI, gpt-4o-mini"},
    ).status_code == 422
