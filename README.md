# gender-by-voice
Celem projektu było stworzenie skryptu w języku Python pozwalającego na rozpoznanie płci osoby mówiącej na nagraniu. Rozpoznanie odbywa się za pomocą algotytmu HPS (Harmonic Product Spectrum). Nagranie głosu w formacie `.wav` przekazywane jest w argumencie wywołania programu. Program zwraca jedną z dwóch liter - **K** jeżeli na nagraniu wykryto kobiecy głos lub **M** w przypadku wykrycia głosu męskiego.

#### Przykładowe wywołanie programu
```
> python main.py test.wav
```
#### Przykładowy wynik
```
> M
```
