function SummaryPanel({
  summaries,
  selectedSlideId,
  onSelectSlide,
  onDownload,
}) {
  return (
    <section className="panel summary-panel">
      <div className="panel-header sticky-header">
        <h2>Slide Summaries</h2>
        <button type="button" className="primary-btn" onClick={onDownload}>
          Download Word Document
        </button>
      </div>

      <div className="summary-list">
        {summaries.length === 0 ? (
          <p className="muted">No summaries available yet.</p>
        ) : (
          summaries.map((slide) => {
            const isActive = slide.slide_id === selectedSlideId;
            return (
              <article
                key={slide.slide_id}
                className={`summary-card ${isActive ? "active" : ""}`}
                onClick={() => onSelectSlide(slide.slide_id)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onSelectSlide(slide.slide_id);
                  }
                }}
                role="button"
                tabIndex={0}
              >
                <div className="summary-meta">
                  <span>Page {slide.page_number}</span>
                  <span>{slide.slide_id}</span>
                </div>
                <h3>{slide.heading || slide.slide_id}</h3>
                <p>{slide.summary}</p>
                <div className="chip-row">
                  {slide.key_terms?.length ? (
                    slide.key_terms.map((term) => (
                      <span key={`${slide.slide_id}-${term}`} className="chip">
                        {term}
                      </span>
                    ))
                  ) : (
                    <span className="muted">No key terms</span>
                  )}
                </div>
              </article>
            );
          })
        )}
      </div>
    </section>
  );
}

export default SummaryPanel;
