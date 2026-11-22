import { useEffect, useRef } from 'react';
import { Upload } from 'lucide-react';
import { testImages } from '../data/images';

/**
 * Selettore dell'immagine di test per il visualizzatore OpenCV.js,
 * con scelta tra le immagini integrate e caricamento di un file locale.
 */
const ImagePicker = ({ selectedId, customLabel, onSelect, id = 'test-image-select' }) => {
  const objectUrlRef = useRef(null);

  useEffect(
    () => () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    },
    [],
  );

  const handlePresetChange = (e) => {
    const img = testImages.find((item) => item.id === e.target.value);
    if (img) onSelect({ id: img.id, src: img.src, name: img.name });
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    onSelect({ id: 'custom', src: url, name: file.name });
    e.target.value = '';
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      <label htmlFor={id} className="sr-only">
        Immagine di test
      </label>
      <select
        id={id}
        value={selectedId}
        onChange={handlePresetChange}
        className="h-8 min-w-0 flex-1 rounded-md border border-line-strong bg-surface px-2.5 text-xs text-ink focus:border-accent focus:outline-none"
      >
        {testImages.map((img) => (
          <option key={img.id} value={img.id}>
            {img.name} — {img.hint}
          </option>
        ))}
        {selectedId === 'custom' && (
          <option value="custom">{customLabel || 'Immagine personalizzata'}</option>
        )}
      </select>
      <label className="btn btn-secondary btn-sm h-8 shrink-0 cursor-pointer px-2.5">
        <Upload size={13} aria-hidden="true" />
        <span>Carica</span>
        <input type="file" accept="image/*" onChange={handleFileChange} className="sr-only" />
      </label>
    </div>
  );
};

export default ImagePicker;
