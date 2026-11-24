import { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { algorithms } from '../data/algorithms';
import { highlightCpp, stripComments } from '../lib/cppHighlight';
import { SYNTAX_COLORS } from '../lib/monacoTheme';

const STORAGE_ALGO = 'imageproc-watch-algo';
const STORAGE_SIZE = 'imageproc-watch-size';
// Moltiplicatori del corpo del codice, ciclati dal pulsante "Aa".
const SIZES = [0.85, 1, 1.2];
const SWIPE_MIN_PX = 50;

const readStored = (key, fallback) => {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
};

const store = (key, value) => {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Senza storage si riparte semplicemente dal primo algoritmo.
  }
};

// La vista è sempre scura: stessi colori del tema scuro dell'editor.
const tokenStyle = (kind) => (kind ? { color: `#${SYNTAX_COLORS.dark[kind]}` } : undefined);

const NavButton = ({ onClick, label, children }) => (
  <button
    type="button"
    onClick={onClick}
    aria-label={label}
    className="flex size-[2.6em] shrink-0 items-center justify-center rounded-full bg-surface text-ink-2 active:bg-line"
  >
    {children}
  </button>
);

/**
 * Vista per smartwatch: solo il codice degli algoritmi, uno per volta,
 * colorato come nell'editor e senza commenti.
 * Schermo tondo e piccolo: tutto è in vw (stessa resa a ogni densità),
 * il codice va a capo invece di scorrere in orizzontale e si cambia
 * algoritmo con le frecce o con uno swipe orizzontale.
 */
const WatchView = () => {
  const [index, setIndex] = useState(() => {
    const id = readStored(STORAGE_ALGO, '');
    return Math.max(0, algorithms.findIndex((a) => a.id === id));
  });
  const [size, setSize] = useState(() => {
    const stored = Number(readStored(STORAGE_SIZE, '1'));
    return SIZES.includes(stored) ? stored : 1;
  });
  const touch = useRef(null);

  const algo = algorithms[index];
  const lines = useMemo(() => stripComments(highlightCpp(algo.codeReference)), [algo]);
  const count = algorithms.length;
  const next = algorithms[(index + 1) % count];

  useEffect(() => {
    store(STORAGE_ALGO, algo.id);
    window.scrollTo(0, 0);
  }, [algo]);

  useEffect(() => {
    store(STORAGE_SIZE, String(size));
  }, [size]);

  const go = (step) => setIndex((i) => (i + step + count) % count);
  const cycleSize = () => setSize((s) => SIZES[(SIZES.indexOf(s) + 1) % SIZES.length]);

  const onTouchStart = (e) => {
    const t = e.touches[0];
    touch.current = { x: t.clientX, y: t.clientY };
  };
  const onTouchEnd = (e) => {
    if (!touch.current) return;
    const t = e.changedTouches[0];
    const dx = t.clientX - touch.current.x;
    const dy = t.clientY - touch.current.y;
    touch.current = null;
    if (Math.abs(dx) >= SWIPE_MIN_PX && Math.abs(dx) > 2 * Math.abs(dy)) go(dx < 0 ? 1 : -1);
  };

  return (
    <div
      data-theme="dark"
      className="min-h-[100dvh] bg-black text-ink"
      style={{ fontSize: '5vw' }}
      onTouchStart={onTouchStart}
      onTouchEnd={onTouchEnd}
    >
      {/* Lo schermo è tondo: margini ampi in alto e in basso, contenuto centrato. */}
      <header className="flex flex-col items-center gap-[0.4em] px-[12%] pb-[0.6em] pt-[9vw] text-center">
        <div className="flex items-center gap-[0.6em]">
          <NavButton onClick={() => go(-1)} label="Algoritmo precedente">
            <ChevronLeft className="size-[1.3em]" aria-hidden="true" />
          </NavButton>
          <span className="font-mono text-[0.8em] text-ink-3" aria-live="polite">
            {index + 1}/{count}
          </span>
          <NavButton onClick={() => go(1)} label="Algoritmo successivo">
            <ChevronRight className="size-[1.3em]" aria-hidden="true" />
          </NavButton>
        </div>
        <h1 className="text-[1.05em] font-semibold leading-tight">{algo.name}</h1>
        <button
          type="button"
          onClick={cycleSize}
          className="rounded-full border border-line px-[0.8em] py-[0.15em] font-mono text-[0.75em] text-ink-2 active:bg-line"
          aria-label="Cambia dimensione del testo"
        >
          Aa {Math.round(size * 100)}%
        </button>
      </header>

      <main>
        <pre
          className="px-[8%] font-mono leading-[1.45]"
          style={{ fontSize: `${0.82 * size}em` }}
          aria-label={`Codice di ${algo.name}`}
        >
          {lines.map((line, i) => (
            // Indentazione dimezzata e rientro sospeso: le righe lunghe vanno a capo
            // allineate al proprio livello invece di ripartire dal bordo.
            <div
              key={i}
              className="min-h-[1.45em] whitespace-pre-wrap break-words"
              style={{ paddingLeft: `${line.indent / 2 + 1}ch`, textIndent: '-1ch' }}
            >
              {line.parts.map((part, j) => (
                <span key={j} style={tokenStyle(part.kind)}>
                  {part.text}
                </span>
              ))}
            </div>
          ))}
        </pre>
      </main>

      <footer className="flex flex-col items-center px-[15%] pb-[16vw] pt-[1.2em]">
        <button
          type="button"
          onClick={() => go(1)}
          className="flex items-center gap-[0.3em] rounded-full bg-surface px-[1em] py-[0.5em] text-[0.85em] text-ink active:bg-line"
        >
          <span className="truncate">{next.name}</span>
          <ChevronRight className="size-[1.1em] shrink-0" aria-hidden="true" />
        </button>
      </footer>
    </div>
  );
};

export default WatchView;
