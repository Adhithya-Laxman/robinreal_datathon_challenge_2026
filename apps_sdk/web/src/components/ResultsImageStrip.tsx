type ListingData = {
  title: string;
  image_urls?: string[] | null;
  hero_image_url?: string | null;
};

type RankedListingResult = {
  listing_id: string;
  listing: ListingData;
};

type ResultsImageStripProps = {
  results: RankedListingResult[];
  selectedId: string | null;
  onSelect: (listingId: string) => void;
};

function firstImageUrl(listing: ListingData): string | null {
  const hero = listing.hero_image_url;
  const urls = listing.image_urls ?? [];
  const candidates = [hero, ...urls].filter((u): u is string => Boolean(u));
  return candidates[0] ?? null;
}

export default function ResultsImageStrip({
  results,
  selectedId,
  onSelect,
}: ResultsImageStripProps) {
  if (!results.length) {
    return null;
  }

  return (
    <div className="results-image-strip" aria-label="Listing photos preview">
      {results.map((result) => {
        const src = firstImageUrl(result.listing);
        if (!src) {
          return null;
        }
        const selected = selectedId === result.listing_id;
        return (
          <button
            key={result.listing_id}
            type="button"
            className={`results-image-strip-item ${selected ? "selected" : ""}`}
            onClick={() => onSelect(result.listing_id)}
            title={result.listing.title}
          >
            <img src={src} alt="" loading="lazy" referrerPolicy="no-referrer" />
          </button>
        );
      })}
    </div>
  );
}
