import lenaSrc from '../assets/lena.png';
import shapesSrc from '../assets/shapes.png';
import checkerSrc from '../assets/checker.png';
import aero1Src from '../assets/samples/aero1.jpg';
import baboonSrc from '../assets/samples/baboon.jpg';
import boardSrc from '../assets/samples/board.jpg';
import buildingSrc from '../assets/samples/building.jpg';
import butterflySrc from '../assets/samples/butterfly.jpg';
import cardsSrc from '../assets/samples/cards.png';
import chickySrc from '../assets/samples/chicky_512.png';
import detectBlobSrc from '../assets/samples/detect_blob.png';
import fruitsSrc from '../assets/samples/fruits.jpg';
import homeSrc from '../assets/samples/home.jpg';
import messiSrc from '../assets/samples/messi5.jpg';
import pic1Src from '../assets/samples/pic1.png';
import pic2Src from '../assets/samples/pic2.png';
import pic3Src from '../assets/samples/pic3.png';
import pic4Src from '../assets/samples/pic4.png';
import pic5Src from '../assets/samples/pic5.png';
import starryNightSrc from '../assets/samples/starry_night.jpg';
import stuffSrc from '../assets/samples/stuff.jpg';
import sudokuSrc from '../assets/samples/sudoku.png';
import gridSrc from '../../../Algoritmi Esame/Harris/grid.jpg';
import smartiesSrc from '../../../Algoritmi Esame/HoughCircles/smarties.png';
import opencvLogoSrc from '../../../Esercizi/showimage/opencv_logo.png';
import sampleGraySrc from '../../../Esercizi/padding di grigi/image.png';

export const testImages = [
  {
    id: 'lena',
    name: 'Lena',
    hint: '512×512 · ritratto classico',
    src: lenaSrc,
    bestFor: ['canny', 'kmeans', 'otsu2k', 'split_merge'],
  },
  {
    id: 'sudoku',
    name: 'Sudoku',
    hint: 'Rette, griglia e alto contrasto',
    src: sudokuSrc,
    bestFor: ['hough_lines', 'canny', 'harris', 'otsu'],
  },
  {
    id: 'building',
    name: 'Edificio',
    hint: 'Angoli architettonici e linee',
    src: buildingSrc,
    bestFor: ['harris', 'hough_lines', 'canny'],
  },
  {
    id: 'grid',
    name: 'Griglia',
    hint: 'Ideale per Harris e Hough Lines',
    src: gridSrc,
    bestFor: ['harris', 'hough_lines'],
  },
  {
    id: 'board',
    name: 'Scacchiera calibrazione',
    hint: 'Angoli netti e rette',
    src: boardSrc,
    bestFor: ['harris', 'hough_lines', 'canny', 'otsu'],
  },
  {
    id: 'smarties',
    name: 'Smarties',
    hint: 'Cerchi colorati e regioni nette',
    src: smartiesSrc,
    bestFor: ['hough_circles', 'kmeans', 'region_growing'],
  },
  {
    id: 'detect_blob',
    name: 'Cerchi e blob',
    hint: 'Cerchi ad alto contrasto',
    src: detectBlobSrc,
    bestFor: ['hough_circles', 'otsu', 'canny', 'region_growing'],
  },
  {
    id: 'stuff',
    name: 'Oggetti vari',
    hint: 'Forme circolari e sogliatura',
    src: stuffSrc,
    bestFor: ['hough_circles', 'otsu', 'region_growing'],
  },
  {
    id: 'fruits',
    name: 'Frutta',
    hint: 'Colori e regioni omogenee',
    src: fruitsSrc,
    bestFor: ['kmeans', 'region_growing', 'split_merge'],
  },
  {
    id: 'butterfly',
    name: 'Farfalla',
    hint: 'Colori vivaci e bordi naturali',
    src: butterflySrc,
    bestFor: ['kmeans', 'canny', 'region_growing', 'otsu2k'],
  },
  {
    id: 'starry_night',
    name: 'Notte stellata',
    hint: 'Palette artistica e texture',
    src: starryNightSrc,
    bestFor: ['kmeans', 'split_merge', 'otsu2k'],
  },
  {
    id: 'baboon',
    name: 'Mandrillo',
    hint: 'Texture ricche e colori',
    src: baboonSrc,
    bestFor: ['kmeans', 'canny', 'split_merge'],
  },
  {
    id: 'chicky',
    name: 'Pulcino',
    hint: '512×512 · soggetto su sfondo morbido',
    src: chickySrc,
    bestFor: ['kmeans', 'otsu', 'otsu2k', 'region_growing'],
  },
  {
    id: 'messi',
    name: 'Calciatore',
    hint: 'Scena sportiva a colori',
    src: messiSrc,
    bestFor: ['kmeans', 'canny', 'region_growing', 'split_merge'],
  },
  {
    id: 'home',
    name: 'Casa',
    hint: 'Bordi, regioni e più livelli',
    src: homeSrc,
    bestFor: ['canny', 'hough_lines', 'otsu2k', 'region_growing', 'split_merge'],
  },
  {
    id: 'cards',
    name: 'Carte da gioco',
    hint: 'Bordi netti, angoli e simboli',
    src: cardsSrc,
    bestFor: ['canny', 'harris', 'hough_lines', 'otsu'],
  },
  {
    id: 'aero1',
    name: 'Vista aerea',
    hint: 'Linee, tetti e regioni',
    src: aero1Src,
    bestFor: ['canny', 'hough_lines', 'split_merge', 'otsu2k'],
  },
  {
    id: 'pic1',
    name: 'Forme geometriche 1',
    hint: 'Oggetti chiari su sfondo scuro',
    src: pic1Src,
    bestFor: ['otsu', 'canny', 'harris', 'hough_lines'],
  },
  {
    id: 'pic2',
    name: 'Scena interna',
    hint: 'Illuminazione e regioni',
    src: pic2Src,
    bestFor: ['canny', 'otsu2k', 'split_merge', 'kmeans'],
  },
  {
    id: 'pic3',
    name: 'Forme e contorni',
    hint: 'Contorni netti e cerchi',
    src: pic3Src,
    bestFor: ['canny', 'hough_circles', 'hough_lines', 'otsu'],
  },
  {
    id: 'pic4',
    name: 'Oggetti su piano',
    hint: 'Segmentazione e bordi',
    src: pic4Src,
    bestFor: ['otsu2k', 'region_growing', 'split_merge', 'canny'],
  },
  {
    id: 'pic5',
    name: 'Pattern geometrico',
    hint: 'Rette, angoli e cerchi',
    src: pic5Src,
    bestFor: ['harris', 'hough_lines', 'hough_circles', 'canny'],
  },
  {
    id: 'shapes',
    name: 'Forme geometriche 2',
    hint: 'Cerchi, rette, angoli e 3 livelli',
    src: shapesSrc,
    bestFor: ['hough_circles', 'hough_lines', 'harris', 'otsu', 'otsu2k', 'region_growing'],
  },
  {
    id: 'checker',
    name: 'Scacchiera sintetica',
    hint: 'Ideale per Harris, Canny e Split & Merge',
    src: checkerSrc,
    bestFor: ['harris', 'canny', 'otsu', 'otsu2k', 'split_merge'],
  },
  {
    id: 'opencv_logo',
    name: 'Logo OpenCV',
    hint: 'Bordi netti e colori',
    src: opencvLogoSrc,
    bestFor: ['canny', 'hough_circles', 'kmeans', 'region_growing'],
  },
  {
    id: 'sample_gray',
    name: 'Campione grigi',
    hint: '640×480 · scala di grigi',
    src: sampleGraySrc,
    bestFor: ['otsu', 'otsu2k'],
  },
];

export const defaultTestImage = testImages[0];

/**
 * Sceglie un'immagine casuale tra quelle ottimizzate per l'algoritmo indicato
 * (evitando, quando possibile, la stessa immagine appena usata).
 */
export const pickImageForAlgo = (algoId, excludeId) => {
  const optimized = testImages.filter((img) => img.bestFor?.includes(algoId));
  const pool = optimized.length > 0 ? optimized : testImages;
  const filtered = pool.length > 1 && excludeId ? pool.filter((img) => img.id !== excludeId) : pool;
  return filtered[Math.floor(Math.random() * filtered.length)];
};
