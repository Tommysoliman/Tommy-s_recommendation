import { useEffect, useRef, useState } from "react";
import {
  getLocalMovies,
  getMovie,
  getRecommendations,
  getTrending,
  searchMovies,
  sendChat,
} from "./api";

function yearOf(movie) {
  return movie.release_date?.slice(0, 4) || "";
}

function MovieCard({ movie, onSelect, compact = false }) {
  return (
    <button className={`movie-card ${compact ? "movie-card-compact" : ""}`} onClick={() => onSelect(movie)} type="button">
      <div className="poster-wrap">
        {movie.poster_url ? (
          <img src={movie.poster_url} alt={`${movie.title} poster`} loading="lazy" />
        ) : (
          <div className="poster-placeholder"><span>✦</span><small>{movie.title}</small></div>
        )}
        {movie.rating != null && <span className="rating-pill">★ {movie.rating.toFixed(1)}</span>}
        {movie.similarity_score != null && (
          <span className="rating-pill match-pill">{Math.round(movie.similarity_score * 100)}% match</span>
        )}
        <div className="poster-hover"><span>View details ↗</span></div>
      </div>
      <div className="movie-card-info">
        <strong>{movie.title}</strong>
        <span>{yearOf(movie) || (movie.rating_source ? "Now streaming" : "ML recommendation")}</span>
        {movie.rating_source && <small className="rating-source">{movie.rating_source} rating</small>}
        {movie.explanation && <p>{movie.explanation}</p>}
      </div>
    </button>
  );
}

function MovieRow({ title, eyebrow, movies, onSelect, emptyMessage }) {
  if (!movies.length) {
    return emptyMessage ? <p className="empty-state">{emptyMessage}</p> : null;
  }
  return (
    <section className="movie-section">
      <div className="section-heading">
        <div><span className="eyebrow">{eyebrow}</span><h2>{title}</h2></div>
        <span className="section-count">{String(movies.length).padStart(2, "0")} films</span>
      </div>
      <div className="movie-grid">
        {movies.map((movie, index) => (
          <MovieCard key={`${movie.movie_id || movie.title}-${index}`} movie={movie} onSelect={onSelect} />
        ))}
      </div>
    </section>
  );
}

export default function App() {
  const [trending, setTrending] = useState([]);
  const [trendLoading, setTrendLoading] = useState(true);
  const [trendError, setTrendError] = useState("");
  const [search, setSearch] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [searchError, setSearchError] = useState("");
  const [searching, setSearching] = useState(false);
  const [mlTitle, setMlTitle] = useState("");
  const [catalog, setCatalog] = useState([]);
  const [catalogError, setCatalogError] = useState("");
  const [mlResults, setMlResults] = useState([]);
  const [mlError, setMlError] = useState("");
  const [mlLoading, setMlLoading] = useState(false);
  const [suggestions, setSuggestions] = useState(false);
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");
  const [sessionId, setSessionId] = useState(() => sessionStorage.getItem("cineverse-session") || "");
  const [selectedMovie, setSelectedMovie] = useState(null);
  const [detailError, setDetailError] = useState("");
  const [detailLoading, setDetailLoading] = useState(false);
  const detailRequestRef = useRef(0);
  const chatEndRef = useRef(null);

  useEffect(() => {
    getTrending()
      .then(setTrending)
      .catch((error) => setTrendError(error.message))
      .finally(() => setTrendLoading(false));
  }, []);

  useEffect(() => {
    if (mlTitle.trim().length < 2) {
      setCatalog([]);
      setCatalogError("");
      return undefined;
    }
    let active = true;
    setCatalogError("");
    const timer = setTimeout(() => {
      getLocalMovies(mlTitle.trim())
        .then((results) => {
          if (active) setCatalog(results);
        })
        .catch((error) => {
          if (active) {
            setCatalog([]);
            setCatalogError(error.message);
          }
        });
    }, 220);
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [mlTitle]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, chatLoading]);

  async function runSearch(event) {
    event.preventDefault();
    if (!search.trim()) return;
    setSearching(true);
    setSearchError("");
    try {
      setSearchResults(await searchMovies(search.trim()));
    } catch (error) {
      setSearchResults([]);
      setSearchError(error.message);
    } finally {
      setSearching(false);
    }
  }

  async function runRecommendations(event) {
    event.preventDefault();
    if (!mlTitle.trim()) return;
    setMlLoading(true);
    setMlError("");
    try {
      const result = await getRecommendations(mlTitle.trim());
      setMlTitle(result.query);
      setSuggestions(false);
      setMlResults(result.recommendations);
    } catch (error) {
      setMlResults([]);
      setMlError(error.message);
    } finally {
      setMlLoading(false);
    }
  }

  async function openMovie(movie) {
    const requestId = ++detailRequestRef.current;
    setSelectedMovie(movie);
    setDetailError("");
    if (!movie.movie_id) {
      setDetailLoading(false);
      return;
    }
    setDetailLoading(true);
    try {
      const details = await getMovie(movie.movie_id);
      if (detailRequestRef.current === requestId) setSelectedMovie(details);
    } catch (error) {
      if (detailRequestRef.current === requestId) setDetailError(error.message);
    } finally {
      if (detailRequestRef.current === requestId) setDetailLoading(false);
    }
  }

  async function submitChat(event) {
    event.preventDefault();
    const message = chatInput.trim();
    if (!message || chatLoading) return;
    setChatInput("");
    setChatError("");
    setMessages((current) => [...current, { role: "user", content: message }]);
    setChatLoading(true);
    try {
      const response = await sendChat(message, sessionId);
      setSessionId(response.session_id);
      sessionStorage.setItem("cineverse-session", response.session_id);
      setMessages((current) => [...current, {
        role: "assistant",
        content: response.message,
        recommendations: response.recommendations,
      }]);
    } catch (error) {
      setChatError(error.message);
    } finally {
      setChatLoading(false);
    }
  }

  function clearChat() {
    setMessages([]);
    setSessionId("");
    sessionStorage.removeItem("cineverse-session");
    setChatError("");
  }

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#home"><span className="brand-mark">T</span><span>Tommy's MovieVerse</span></a>
        <nav aria-label="Main navigation">
          <a href="#discover">Discover</a><a href="#classic-ml">The classics</a><a href="#ai-assistant">AI assistant</a>
        </nav>
        <a className="topbar-cta" href="#ai-assistant">Find a film <span>↗</span></a>
      </header>

      <section className="hero" id="home">
        <div className="hero-backdrop" aria-hidden="true" />
        <div className="hero-content">
          <div className="hero-kicker"><span className="live-dot" /> YOUR NEXT FAVORITE IS OUT THERE</div>
          <h1>Discover your<br /><em>next favorite</em> movie.</h1>
          <p className="hero-subtitle">A little movie magic, powered by AI and a love of great stories.</p>
          <form className="hero-search" onSubmit={runSearch}>
            <span className="search-symbol">⌕</span>
            <input aria-label="Search movies" placeholder="Search a title, mood, or moment..." value={search} onChange={(event) => setSearch(event.target.value)} />
            <button type="submit" disabled={searching}>{searching ? "Searching..." : "Search films"} <span>↗</span></button>
          </form>
          <div className="hero-footnote"><span>✦</span> Real movie data from TMDB · Recommendations with a human touch</div>
        </div>
        <div className="hero-index"><span>01</span><i /> DISCOVER</div>
      </section>

      {(searchResults.length > 0 || searchError) && (
        <div className="page-content search-results">
          {searchError && <div className="notice error-notice">{searchError}</div>}
          <MovieRow title={`Results for “${search}”`} eyebrow="YOUR SEARCH" movies={searchResults} onSelect={openMovie} emptyMessage={searching ? "" : "No matching movies found."} />
        </div>
      )}

      <div className="page-content">
        <section className="intro-row" id="discover">
          <div><span className="eyebrow">A WORLD OF STORIES</span><h2>Tonight's <em>inspiration</em></h2></div>
          <p>Handpicked from what's moving audiences right now.</p>
        </section>
        {trendError && <div className="notice">{trendError} Add a TMDB key to load live discoveries.</div>}
        {trendLoading && <div className="loading-cards" aria-label="Loading trending movies">{Array.from({ length: 5 }, (_, index) => <span key={index} />)}</div>}
        <MovieRow title="Trending this week" eyebrow="IN THE SPOTLIGHT" movies={trending} onSelect={openMovie} />

        <section className="ml-panel" id="classic-ml">
          <div className="ml-copy">
            <span className="eyebrow">THE CLASSIC ALGORITHM</span>
            <h2>Find the next film<br />in your <em>orbit.</em></h2>
            <p>Our original machine-learning model compares the story DNA of 4,800+ films using CountVectorizer and cosine similarity. No black box. Just good taste, in math.</p>
            <div className="ml-stats"><span><strong>4.8k</strong> films indexed</span><span><strong>10</strong> picks per search</span></div>
          </div>
          <div className="ml-form-card">
            <div className="ml-card-top"><span className="mini-star">✳</span><span>SIMILARITY ENGINE <i>● LIVE</i></span></div>
            <h3>Start with a film you love.</h3>
            <p>We'll look for stories with a similar feel.</p>
            <form className="ml-form" onSubmit={runRecommendations}>
              <div className="suggest-wrap">
                <input aria-label="Movie title for recommendations" placeholder="Try “Interstellar”..." value={mlTitle} onFocus={() => setSuggestions(true)} onChange={(event) => { setMlTitle(event.target.value); setSuggestions(true); }} />
                {suggestions && catalog.length > 0 && (
                  <div className="suggestions">
                    {catalog.slice(0, 6).map((movie) => <button type="button" key={movie.movie_id} onClick={() => { setMlTitle(movie.title); setSuggestions(false); }}>{movie.title}</button>)}
                  </div>
                )}
              </div>
              <button type="submit" disabled={mlLoading}>{mlLoading ? "Finding films..." : "Find similar films"} <span>→</span></button>
            </form>
            {catalogError && <p className="inline-error">{catalogError}</p>}
            {mlError && <p className="inline-error">{mlError}</p>}
            {mlResults.length > 0 && <p className="result-note">Showing the closest matches for <strong>{mlTitle}</strong></p>}
          </div>
        </section>
        <MovieRow title="A similar kind of magic" eyebrow="ML RECOMMENDATIONS" movies={mlResults} onSelect={openMovie} />

        <section className="assistant-section" id="ai-assistant">
          <div className="assistant-copy">
            <span className="eyebrow">YOUR PERSONAL FILM SCOUT</span>
            <h2>Tell us the <em>feeling.</em></h2>
            <p>Moody and mysterious? Something to laugh at together? Just say the word. Our AI looks up real films and explains why they might be your kind of story.</p>
            <div className="assistant-tags"><span>✦ Natural language</span><span>✦ Real movie facts</span><span>✦ Remembers this chat</span></div>
          </div>
          <div className="chat-card">
            <div className="chat-header">
              <div className="assistant-avatar">✳</div><div><strong>Film scout</strong><span><i className="live-dot" /> READY WHEN YOU ARE</span></div>
              <button className="clear-chat" type="button" title="Start a new conversation" onClick={clearChat}>↻</button>
            </div>
            <div className="chat-messages">
              {messages.length === 0 && <div className="assistant-message"><span className="message-spark">✳</span><div><p>Hi, I'm your movie scout. Tell me what kind of story you're in the mood for, and I'll find a few real films worth your time.</p><div className="prompt-chips">{["A clever thriller", "Something like Interstellar", "Funny date-night movies"].map((prompt) => <button type="button" key={prompt} onClick={() => setChatInput(prompt)}>{prompt} ↗</button>)}</div></div></div>}
              {messages.map((message, index) => (
                <div className={`message ${message.role}`} key={`${message.role}-${index}`}>
                  {message.role === "assistant" && <span className="message-spark">✳</span>}
                  <div><p>{message.content}</p>
                    {message.recommendations?.length > 0 && <div className="chat-recommendations">{message.recommendations.map((movie, movieIndex) => <MovieCard key={`${movie.movie_id || movie.title}-${movieIndex}`} movie={movie} onSelect={openMovie} compact />)}</div>}
                  </div>
                </div>
              ))}
              {chatLoading && <div className="message assistant"><span className="message-spark">✳</span><div className="typing-indicator"><i /><i /><i /></div></div>}
              <div ref={chatEndRef} />
            </div>
            {chatError && <div className="chat-error">{chatError}</div>}
            <form className="chat-input" onSubmit={submitChat}>
              <input aria-label="Describe what you want to watch" placeholder="Describe what you're in the mood for..." value={chatInput} onChange={(event) => setChatInput(event.target.value)} maxLength={2000} />
              <button type="submit" aria-label="Send message" disabled={chatLoading || !chatInput.trim()}>↑</button>
            </form>
            <div className="chat-footnote">Movie details and ratings are provided by TMDB. Ratings are not IMDb ratings.</div>
          </div>
        </section>
      </div>

      <footer><a className="brand" href="#home"><span className="brand-mark">T</span><span>Tommy's MovieVerse</span></a><span>Made for people who love a good story.</span><a href="#home">BACK TO TOP ↑</a></footer>

      {selectedMovie && (
        <div className="modal-backdrop" role="presentation" onClick={() => { detailRequestRef.current += 1; setDetailLoading(false); setSelectedMovie(null); }}>
          <section className="detail-modal" role="dialog" aria-modal="true" aria-label={`${selectedMovie.title} details`} onClick={(event) => event.stopPropagation()}>
            <button className="modal-close" type="button" onClick={() => { detailRequestRef.current += 1; setDetailLoading(false); setSelectedMovie(null); }} aria-label="Close details">×</button>
            {selectedMovie.poster_url && <img className="detail-poster" src={selectedMovie.poster_url} alt="" />}
            <div className="detail-content"><span className="eyebrow">MOVIE PROFILE</span><h2>{selectedMovie.title}</h2>
              {detailLoading && <p>Loading the full story...</p>}
              {detailError && <p className="inline-error">{detailError}</p>}
              <div className="detail-meta">{yearOf(selectedMovie) && <span>{yearOf(selectedMovie)}</span>}{selectedMovie.rating != null && <span>★ {selectedMovie.rating.toFixed(1)} {selectedMovie.rating_source || ""}</span>}{selectedMovie.genres?.map((genre) => <span key={genre}>{genre}</span>)}</div>
              <p className="detail-overview">{selectedMovie.overview || "A little more mystery to discover."}</p>
              {!detailLoading && selectedMovie.movie_id && (
                <div className="where-to-watch">
                  <strong>Where to watch{selectedMovie.where_to_watch ? ` (${selectedMovie.where_to_watch.region})` : ""}</strong>
                  {[["stream", "Stream"], ["rent", "Rent"], ["buy", "Buy"]].map(([key, label]) => selectedMovie.where_to_watch?.[key]?.length > 0 && (
                    <p key={key}><span>{label}</span>{selectedMovie.where_to_watch[key].join(" · ")}</p>
                  ))}
                  {!selectedMovie.where_to_watch && <p className="muted">No streaming info available in your region.</p>}
                  {selectedMovie.where_to_watch?.link && <a className="trailer-link" href={selectedMovie.where_to_watch.link} target="_blank" rel="noreferrer">See all options ↗</a>}
                  <small>Streaming data by JustWatch via TMDB.</small>
                </div>
              )}
              {selectedMovie.trailer_url && <a className="trailer-link" href={selectedMovie.trailer_url} target="_blank" rel="noreferrer">Watch trailer ↗</a>}
              {selectedMovie.rating_source && <small className="detail-source">Rating source: {selectedMovie.rating_source}. TMDB ratings are distinct from IMDb ratings.</small>}
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
