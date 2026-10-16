const API_BASE = (
  window.RUNTIME_CONFIG?.VITE_API_URL ||
  import.meta.env.VITE_API_URL ||
  "http://localhost:8000"
).replace(/\/+$/, "");

async function request(path, options) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch {
    throw new Error("Can't reach the movie service. Check your connection and try again.");
  }
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error("The movie service returned an invalid response.");
  }
  if (!response.ok) {
    throw new Error(payload.detail || `The movie service returned an error (${response.status}).`);
  }
  return payload;
}

export const getTrending = () => request("/api/movies/trending");
export const searchMovies = (query) => request(`/api/movies/search?q=${encodeURIComponent(query)}`);
export const getMovie = (id) => request(`/api/movies/${encodeURIComponent(id)}`);
export const getLocalMovies = (query) => request(`/api/movies/catalog?q=${encodeURIComponent(query)}`);
export const getRecommendations = (title) =>
  request(`/api/recommendations/${encodeURIComponent(title)}`);
export const sendChat = (message, sessionId) =>
  request("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId || null }),
  });
