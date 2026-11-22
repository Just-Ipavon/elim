<div align="center">

# Elaborazione delle Immagini

Algoritmi di visione artificiale in OpenCV C++ e applicazione web interattiva per studiarli, riscriverli e testarli.

<p>
  <a href="https://just-ipavon.github.io/elim/">
    <img src="https://img.shields.io/badge/GitHub%20Pages-Sito%20Live-2ea44f?style=for-the-badge&logo=githubpages&logoColor=white" alt="Sito Live su GitHub Pages" />
  </a>
  <a href="https://github.com/Just-Ipavon/elim/actions/workflows/deploy.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/Just-Ipavon/elim/deploy.yml?branch=main&style=for-the-badge&logo=githubactions&logoColor=white&label=Deploy" alt="Stato Deploy" />
  </a>
  <img src="https://img.shields.io/badge/OpenCV-C%2B%2B-00599C?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV C++" />
  <img src="https://img.shields.io/badge/React-Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="React e Vite" />
</p>

<p>
  <a href="https://just-ipavon.github.io/elim/">
    <img src="https://img.shields.io/badge/Apri_il_sito-just--ipavon.github.io%2Felim-c2410c?style=for-the-badge" alt="Apri il sito" />
  </a>
</p>

</div>

---

## Sito web

L'applicazione web in [`elimsite/`](elimsite/) permette di studiare i 9 algoritmi passo per passo, riscriverli nell'editor C++ con verifica automatica, simulare la prova d'esame a tempo ed eseguire il visualizzatore OpenCV.js su diverse immagini di test o caricate dal proprio dispositivo.

<p align="center">
  <a href="https://just-ipavon.github.io/elim/">
    <img src="https://img.shields.io/badge/Vai_al_sito_live-%E2%86%92-2ea44f?style=for-the-badge&logo=githubpages&logoColor=white" alt="Vai al sito live" />
  </a>
</p>

<p align="center">
  <a href="https://just-ipavon.github.io/elim/">
    <img src="docs/site-preview.png" alt="Anteprima del sito Elaborazione delle Immagini" width="820" />
  </a>
</p>

## Struttura del repository

- **`Algoritmi Esame/`** — Implementazioni C++ di riferimento, immagini di test e istruzioni di compilazione (`compila.txt`).
- **`Esercizi/`** — Esercizi introduttivi con OpenCV.
- **`elimsite/`** — Applicazione web (React + Vite + Monaco Editor + OpenCV.js) pubblicata su [just-ipavon.github.io/elim](https://just-ipavon.github.io/elim/). I riferimenti mostrati sul sito leggono direttamente i file `.cpp` presenti in `Algoritmi Esame/`.

## Algoritmi inclusi

| Algoritmo | File sorgente | Descrizione |
| --- | --- | --- |
| **Canny** | [`canny.cpp`](Algoritmi%20Esame/canny/canny.cpp) | Rilevamento dei bordi con blur gaussiano, Sobel, NMS e isteresi. |
| **Harris** | [`Harris.cpp`](Algoritmi%20Esame/Harris/Harris.cpp) | Rilevamento degli angoli tramite matrice di struttura e risposta $R$. |
| **Hough Circles** | [`houghCircles.cpp`](Algoritmi%20Esame/HoughCircles/houghCircles.cpp) | Trasformata di Hough per l'individuazione di circonferenze. |
| **Hough Lines** | [`houghLines.cpp`](Algoritmi%20Esame/HoughLines/houghLines.cpp) | Trasformata di Hough in coordinate polari $(\rho, \theta)$ per le rette. |
| **K-means** | [`kmeans.cpp`](Algoritmi%20Esame/K-means/kmeans.cpp) | Segmentazione tramite clustering K-means sui colori. |
| **Otsu** | [`otsu.cpp`](Algoritmi%20Esame/Otsu/otsu.cpp) | Sogliatura binaria automatica che massimizza la varianza inter-classe. |
| **Otsu 2k** | [`otsu2k.cpp`](Algoritmi%20Esame/Otsu2k/otsu2k.cpp) | Estensione di Otsu a due soglie (tre classi di intensità). |
| **Region Growing** | [`regionGrowing.cpp`](Algoritmi%20Esame/Region%20Growing/regionGrowing.cpp) | Segmentazione per accrescimento di regioni da pixel seme. |
| **Split and Merge** | [`splitAndMerge.cpp`](Algoritmi%20Esame/split%20and%20merge/splitAndMerge.cpp) | Segmentazione quadtree con fase di divisione e fusione. |

## Esecuzione in locale

### Algoritmi C++
Ogni cartella in `Algoritmi Esame/` contiene un file `compila.txt` con il comando `g++` per OpenCV 5 e un'immagine di test.

### Sito web (`elimsite`)
```bash
cd elimsite
npm ci
npm run dev
```
