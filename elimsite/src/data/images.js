import lenaSrc from '../assets/lena.png';
import shapesSrc from '../assets/shapes.png';
import checkerSrc from '../assets/checker.png';
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
  },
  {
    id: 'grid',
    name: 'Griglia',
    hint: 'Ideale per Harris e Hough Lines',
    src: gridSrc,
  },
  {
    id: 'smarties',
    name: 'Smarties',
    hint: 'Ideale per Hough Circles, K-means e Region Growing',
    src: smartiesSrc,
  },
  {
    id: 'shapes',
    name: 'Forme geometriche',
    hint: 'Cerchi, rette, angoli e 3 livelli',
    src: shapesSrc,
  },
  {
    id: 'checker',
    name: 'Scacchiera',
    hint: 'Ideale per Harris, Canny e Split & Merge',
    src: checkerSrc,
  },
  {
    id: 'opencv_logo',
    name: 'Logo OpenCV',
    hint: 'Bordi netti e colori',
    src: opencvLogoSrc,
  },
  {
    id: 'sample_gray',
    name: 'Campione grigi',
    hint: '640×480 · scala di grigi',
    src: sampleGraySrc,
  },
];

export const defaultTestImage = testImages[0];
