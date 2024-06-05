
Cechy:
- proste i zrozumiałe dla przeciętnego odbiorcy, tzn. brak żargonu programisty, analogie, kod w Pythonia tylko dla zobrazowania
- interaktywne, tzn. pytania do publiczności, 


## 1. Wstęp
- Kim jestem? O czym będę mówił?
  - O czym:
    - o uczeniu maszynowym, przede wszystkim o sieciach neuronowych

## 2. Czym jest uczenie maszynowe?
- Zanim przejdziemy do samej definicji, to weźmy pod uwagę hipotetyczny problem:
- Jesteśmy w niedalekiej przyszłości, w której udało nam się wylądować na Marsie
  - Jedynym źródłem pokarmu na tej planecie są owoce, lecz niektóre z nich są trujące
    - *Pokazać obrazki owoców, wyglądających jak pianki, uroczych z uśmiechem - można Amelkę poprosić o rysunek*
  - Zauważamy, że te owoce wyróżniają się od siebie długością kolców oraz wielkością kropek
  - Jak rozróżnić te trujące od tych, które są jadalne?
    - `Pytanie do publiczności`
  - Jednym ze sposobów na rozróżnienie tych owoców jest zaangażowanie wielu odważnych testerów, którzy spróbją wielu owoców i zobaczą, które z nich są trujące
  - Następnie, możemy zebrać dane na temat tych owoców, i nałożyć na graficzne wykresy, które będą przedstawiać zależności między cechami tych owoców
    - *Pokazać wykres, na którym będą przedstawione dane owoców, zaznaczone różnymi kolorami, w zależności od tego, czy są trujące, czy jadalne*
  - Jeżeli nie będzie widocznej zależności między tymi danymi, to prawdopodobnie te cechy nie mają wiele wspólnego z jadalnością tych owoców
    - *Losowe punkty*
  - Jednak jeżeli zauważymy, że dane są bardziej zorganizowane, to możemy próbować odgadywać na kolejnych robakach, które z nich są trujące, a które jadalne
    - *np. linia prosta widoczna na graficznym wykresie*
  - Mając taki wykres, możemy nałożyć na niego funkcję, która będzie rozdzielała te owoce na te trujące i te jadalne, tzw. funkcję decyzyjną
    - W ten sposób, możemy zautomatyzować proces rozróżniania tych owoców i przewidywać, które z nich są trujące, a które jadalne
- Do tego m.in. służy uczenie maszynowe, mając otwarty problem, i wiele danych, możemy nauczyć maszynę rozróżniać te dane, i przewidywać wyniki
- Definicja: Uczenie maszynowe to dziedzina sztucznej inteligencji, która pozwala komputerom na naukę bez wyraźnego programowania
  - W skrócie: komputer sam się uczy, na podstawie danych, które dostaje

## 3. Jak można rozwiązać problem?
- Jednym ze sposobów na rozwiązanie wspomianego problemu jest wykorzystanie sieci neuronowych
  - To reprezentacja biologicznych neuronów dla komputera.
  - Pojedynczy neuron skałada się z wielu wejść, które mają pewną wartość, niektóre są `ważniejsze` od innych, oraz mają tylko jedno wyjście.
  - Do tego możemy powiedzieć, że mają one funkcję aktywacyjną. Tzn. jeżeli neuron jest wystarczająco nastymulowany, to dopiero wtedy przekazuje dalej sygnał.
  - W dodatku niektóre neurony mogą być ważniejsze od innych, tzn. nie potrzebują dużej stymulacji, by przekazać informację.
  - Mając te 4 elementy (wiele wejść o różnych `wagach`, jedno wyjście, funkcja aktywacyjna oraz `ważność` samego neuronu), możemy odzwierciedlić taki neuron w komputerze
- Jednak sam neuron niewiele może zdziałać wobec bardziej złożonych problemów, takich jak rozponawanie zwierząt na zdjęciu.
  - By stworzyć odpowiednie narzędzie, trzeba będzie zaangażować całe sieci neuronów
  - Taką jedną warstwę w sieci, możemy przedstawić jako jeden rząd neuronów, które są połączone w pełni z każdym wejściem.
  - Następnie robimy kilka takich warstw, aby uzyskać sieć neuronową

## 4. Jak nauczyć sieć?
- Teraz mamy już sieć neuronową ale co tego, jak jej nie nauczymy? Aktualnie podaje ona losowe wyniki, czego nie chcemy
  - W jaki sposób można nauczyć sieć?
  - Zastanówmy się, co dokładnie chcemy zoptymalizować w tej sieci. 
    - Chcemy, aby wyniki, które otrzymujemy, były jak najbardziej zbliżone do oczekiwanych
  - By określić jak dobrze działa nasza sieć, musimy zdefiniować funkcję kosztu, która będzie określała, jak bardzo wyniki naszej sieci różnią się od oczekiwanych
    - Wyniki bardzo odbiegające od oczekiwanych, będą miały wysoki koszt, a wyniki zbliżone do oczekiwanych, będą miały niski koszt
  - Teraz, mając zdefiniowaną metrykę wydajności, możemy zacząć zastanwiać się jak możemy zoptymalizować naszą sieć
  - Mówiliśmy, że neuron ma `wagi połączeń`, `ważność`, funkcję aktywacyjną, wiele wejść i jedno wyjście.
  - To czy ten neuron 'zapali się' zależy tylko od wejść, ich wag oraz ważności neuronu.
  - Samych wejść nie będziemy modifikować, są to np. zdjęcia, lub cechy owoców, które chcemy rozróżnić
  - Mamy 2 cechy do dyspozycji:
    - Wagi połączeń między neuronami
    - Ważność neuronów
  - Wagi połączeń między neuronami są odpowiedzialne za to, jak bardzo dany neuron jest zależny od innych neuronów, 
    - *Przykład: jeżeli neuron odpowiedzialny za rozpoznawanie kota, jest mocno połączony z neuronem odpowiedzialnym za rozpoznawanie uszu, to jeżeli sieć dostanie zdjęcie kota bez uszu, to nie będzie w stanie rozpoznać kota*
  - Ważność neuronów jest odpowiedzialna za to, jak bardzo dany neuron jest ważny dla całej sieci
    - *Przykład: jeżeli neuron odpowiedzialny za rozpoznawanie kota, jest ważniejszy od neuronu odpowiedzialnego za rozpoznawanie psa, to sieć będzie bardziej skłonna do rozpoznawania kotów niż psów*
  - Więc obie te cechy będą wpływać na wynik naszej sieci, co wpływa na koszt, który chcemy zminimalizować
  - Nie zagłębiając się w szczegóły, wykorzystywany jest do tego algorytm wstecznej propagacji (backpropagation)
  - W skrócie: 
      - *Zajebisty schemat*
    - Podajemy sieci dane oraz oczekiwane wyniki, tzn. podając zdjęcie kota, oczekujemy, że sieć powie, że to kot.
    - Następnie dane przechodzą przez sieć, i otrzymujemy wyniki
    - Sprawdzamy, jak bardzo wyniki różnią się od oczekiwanych, czyli koszt tej sieci
    - Na podstawie tych różnic, modyfikujemy wagi połączeń w sieci, tak aby wyniki były bardziej zbliżone do oczekiwanych

## 5. Uczenie
- Mając zdolną do uczenia się sieć, możemy ją wykorzystać do rozwiązania problemu z robakami na Marsie
  - *Animacja z punktami i wyznaczaniem lini decyzyjnej*
- Przejdźmy do bardziej praktycznego przykładu
  - Rozpoznawanie ręcznie pisanych cyfr, użyjemy dokładnie tego samego podejścia co z robakami, tzn. zbieramy dane i uczymy sieć
    - *Pokazanie zestawu MNIST*
  - Nauczmy w takim razie sieć na tym zbiorze
    - *Jakaś animacja z uczeniem*
  - Po nauczeniu sieci, możemy ją wykorzystać do rozpoznawania cyfr
    - *Tutaj używam rzeczywistego przykładu z siecią trenowaną na podstawowym zbiorze*
      - Pokazuję jak rysuję cyfrę, a sieć ją nie rozpoznaje (bo nie jest wyśrodkowana)
      - `Pytanie do publiczności`: 
        - Dlaczego sieć nie rozpoznaje tej cyfry?
      - Odpowiedź to wada zbioru MNIST, tzn. wszystkie cyfry są wyśrodkowane, a ta nie jest
    - Prostym rozwiązaniem tej wady, będzie dodanie losowego przesunięcia do cyfry, by nie była wyśrodkowana
      - *Animacja z pokazaniem tych cyfr*
      - *Uczenie na nowym zbiorze*
    - Żywo pokazać jak sieć rozpoznaje cyfry
      - *Pokazanie jak rysuję cyfrę, a sieć ją rozpoznaje*
    - Teraz mamy inny problem, jeżeli cyfra będzie pod kątem, to sieć nie będzie w stanie jej rozpoznać
      - *Pokazanie jak rysuję cyfrę pod kątem, a sieć nie rozpoznaje*
    - Zatem, powtórzmy proces z dodaniem losowego obrotu cyfry
      - *Animacja z pokazaniem tych cyfr*
      - *Uczenie na nowym zbiorze*
    - Żywo pokazać jak sieć rozpoznaje cyfry
      - *Pokazanie jak rysuję cyfrę pod kątem, a sieć ją rozpoznaje*

## 6. Podsumowanie
- Podsumowując, nauczyliśmy się czym jest uczenie maszynowe, jak działa sieć neuronowa oraz jak można ją nauczyć.
- Warto zauważyć, że sieci neuronowe są bardzo potężnym narzędziem, które można wykorzystać do zautomatyzowania wielu procesów
- Jednak, by sieć działała poprawnie, trzeba ja odpowiednio nauczyć, przedewszystkim na opdowiednio dobranych danych.
- Dziękuję za uwagę, jeżeli macie jakieś pytania, to chętnie na nie odpowiem