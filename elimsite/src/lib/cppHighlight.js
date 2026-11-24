// Evidenziatore C++ leggero per le viste senza Monaco (es. smartwatch).
// Ricalca le regole del tokenizer C++ di Monaco, così i colori coincidono con l'editor.

// Parole chiave riconosciute dal linguaggio C++ di Monaco.
const KEYWORDS = new Set(
  (
    'abstract amp array auto bool break case catch char class const constexpr const_cast ' +
    'continue cpu decltype default delegate delete do double dynamic_cast each else enum ' +
    'event explicit export extern false final finally float for friend gcnew generic goto ' +
    'if in initonly inline int interface interior_ptr internal literal long mutable ' +
    'namespace new noexcept nullptr __nullptr operator override partial pascal pin_ptr ' +
    'private property protected public ref register reinterpret_cast restrict return ' +
    'safe_cast sealed short signed sizeof static static_assert static_cast struct switch ' +
    'template this thread_local throw tile_static true try typedef typeid typename union ' +
    'unsigned using value virtual void volatile wchar_t where while'
  ).split(' '),
);

const RULES = [
  [/^\s+/, null],
  [/^[A-Za-z_]\w*/, (m) => (KEYWORDS.has(m) ? 'keyword' : null)],
  [/^"(?:[^"\\]|\\.)*"?/, 'string'],
  [/^'(?:[^'\\]|\\.)*'?/, 'string'],
  [/^0[xX][0-9a-fA-F']+[uUlL]*/, 'number'],
  [/^(?:\d[\d']*\.?[\d']*|\.\d[\d']*)(?:[eE][-+]?\d+)?[fFlLuU]*/, 'number'],
  [/^[^\sA-Za-z_0-9"'/]+|^\//, null],
];

// Tokenizza il codice di una riga (senza commenti).
const tokenizeCode = (text, out) => {
  let rest = text;
  while (rest) {
    for (const [re, kind] of RULES) {
      const m = rest.match(re);
      if (!m) continue;
      const k = typeof kind === 'function' ? kind(m[0]) : kind;
      out.push({ text: m[0], kind: k });
      rest = rest.slice(m[0].length);
      break;
    }
  }
};

// Direttive del preprocessore: "#include" come parola chiave, il percorso come stringa.
const tokenizeDirective = (text, out) => {
  const inc = text.match(/^(#\s*include)(\s*)(<[^<>]*>|"[^"]*")?(.*)$/);
  if (inc) {
    out.push({ text: inc[1], kind: 'keyword' });
    if (inc[2]) out.push({ text: inc[2], kind: null });
    if (inc[3]) {
      const open = inc[3][0];
      if (open === '<') {
        out.push({ text: '<', kind: 'keyword' });
        out.push({ text: inc[3].slice(1, -1), kind: 'string' });
        out.push({ text: '>', kind: 'keyword' });
      } else {
        out.push({ text: inc[3], kind: 'string' });
      }
    }
    if (inc[4]) tokenizeCode(inc[4], out);
    return;
  }
  const dir = text.match(/^#\s*\w*/)[0];
  out.push({ text: dir, kind: 'keyword' });
  tokenizeCode(text.slice(dir.length), out);
};

/**
 * Divide il codice in righe con indentazione e token colorabili.
 * Ogni token ha `kind`: 'keyword' | 'string' | 'number' | 'comment' | null.
 */
export const highlightCpp = (code) => {
  let inBlock = false;
  return code.split('\n').map((raw) => {
    const indent = raw.match(/^ */)[0].length;
    const text = raw.slice(indent);
    const parts = [];
    let rest = text;
    while (rest) {
      if (inBlock) {
        const end = rest.indexOf('*/');
        const cut = end === -1 ? rest.length : end + 2;
        parts.push({ text: rest.slice(0, cut), kind: 'comment' });
        rest = rest.slice(cut);
        if (end !== -1) inBlock = false;
        continue;
      }
      // Primo inizio di commento fuori dalle stringhe.
      let start = -1;
      for (let i = 0, quote = null; i < rest.length; i++) {
        const c = rest[i];
        if (quote) {
          if (c === '\\') i++;
          else if (c === quote) quote = null;
        } else if (c === '"' || c === "'") quote = c;
        else if (c === '/' && (rest[i + 1] === '/' || rest[i + 1] === '*')) {
          start = i;
          break;
        }
      }
      const code = start === -1 ? rest : rest.slice(0, start);
      if (code) {
        if (parts.length === 0 && code.startsWith('#')) tokenizeDirective(code, parts);
        else tokenizeCode(code, parts);
      }
      if (start === -1) break;
      if (rest[start + 1] === '/') {
        parts.push({ text: rest.slice(start), kind: 'comment' });
        break;
      }
      parts.push({ text: '/*', kind: 'comment' });
      inBlock = true;
      rest = rest.slice(start + 2);
    }
    return { indent, parts };
  });
};

/**
 * Toglie i commenti dalle righe di highlightCpp: le righe di solo commento spariscono,
 * gli spazi rimasti a fine riga vengono rimossi e le righe vuote consecutive ridotte a una.
 */
export const stripComments = (lines) => {
  const out = [];
  for (const line of lines) {
    const hadComment = line.parts.some((p) => p.kind === 'comment');
    const parts = line.parts.filter((p) => p.kind !== 'comment');
    while (parts.length && !parts[parts.length - 1].text.trim()) parts.pop();
    if (parts.length) {
      const last = parts[parts.length - 1];
      parts[parts.length - 1] = { ...last, text: last.text.replace(/\s+$/, '') };
    }
    const empty = parts.length === 0;
    if (empty && hadComment) continue;
    if (empty && (out.length === 0 || out[out.length - 1].parts.length === 0)) continue;
    out.push({ indent: empty ? 0 : line.indent, parts });
  }
  while (out.length && out[out.length - 1].parts.length === 0) out.pop();
  return out;
};
