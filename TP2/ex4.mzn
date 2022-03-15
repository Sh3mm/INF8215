% Antoine ROBILLARD (1899718)
% Rachel MEYER (1903149)

%-2p Quoridor tournament-

include "globals.mzn";
include "factorial.mzn";

int: MAX_T = 30;%nombre maximum de creneaux
int: G = 3;%nombre de tables de tournoi
int: P = 13;%nombre de joueurs

var 1..MAX_T: T;%temps a minimiser

array[1..MAX_T,1..G,1..2] of var 0..P: x;

array[1..P, 1..MAX_T] of var 0..G: tables;
array[1..P, 1..P] of var int: againsts;

% minimum (nCr / G où n = P et r = 2) = 26 parties max 30
var int: fac_P; constraint factorial(P, fac_P);
var int: fac_2; constraint factorial(2, fac_2);
var int: fac_P_m_2; constraint factorial(P-2, fac_P_m_2);

constraint T >= fac_P div (G * fac_2 * fac_P_m_2) /\ T <= MAX_T;

% horaire P1
constraint tables[1, 1..17] == [1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1];
constraint forall(i in 1..17)(
  if (i mod 3) != 0 then tables[(i+1)-(i div 3), i] == 1 endif
);

% s'il y a une partie, 2 joueurs doivent jouer
constraint forall(i in 1..T, j in 1..G)(if j in tables[1..P, i] then count(tables[1..P, i], j) == 2 endif);

% chaque joueur doit avoir rencontré chaque autre joueur en match
constraint forall(i in 1..P, j in 1..P where i != j)(
  1 == sum(k in 1..T where tables[i, k] == tables[j, k] /\ tables[i, k] != 0)(1)
);

% au moins une journée de repos après deux match
constraint forall(i in 1..P, j in 1..(T - 2))(
  if tables[i, j] != 0 /\ tables[i, j+1] != 0 then tables[i, j + 2] == 0 endif
);

% au moins une journée de repos après deux match
constraint forall(i in (T + 1)..MAX_T)(
  tables[1..P, i] == [0,0,0,0,0,0,0,0,0,0,0,0,0]
);

solve minimize T;

output
["Schedule ("++show(T)++" time slots)\n"]
++
[
  "\n"++if length([k | k in 1..P where tables[k, i] == 1]) != 0 then show([k | k in 1..P where tables[k, i] == 1]) else "------" endif++"\t\t"++
  if length([k | k in 1..P where tables[k, i] == 2]) != 0 then show([k | k in 1..P where tables[k, i] == 2]) else "------" endif++"\t\t"++
  if length([k | k in 1..P where tables[k, i] == 3]) != 0 then show([k | k in 1..P where tables[k, i] == 3]) else "------" endif
  | i in 1..MAX_T

]

% -Réponse-
% Schedule (26 time slots) 

% [1,  2 ]	[6,  9 ]	[7,  12]
% [1,  3 ]	[7,  11]	[12, 13]
% [6,  11]	[2,  9 ]	[8,  13]
% [1,  4 ]	[7,  10]	[3,  8 ]
% [1,  5 ]	[9,  11]	[7,  13]
% [5,  6 ]	[3,  12]	[2,  4 ]
% [1,  6 ]	[7,  8 ]	[11, 12]
% [1,  7 ]	[3,  4 ]	[2,  5 ]
% [10, 12] [9,  13]	[6,  8 ]
% [1,  8 ]	[2,  3 ]	[10, 11]
% [1,  9 ]	[11, 13]	[5,  7 ]
% [8,  9 ]	[2,  10]	[6,  12]
% [1,  10]	[4,  6 ]	[5,  11]
% [1,  11]	[5,  12]	[4,  13]
% [10, 13]	[7,  9 ]	[2,  8 ]
% [1,  12]	[2,  6 ]	[4,  5 ]
% [1,  13]	[5,  8 ]	[3,  9 ]
% [3,  11]	[4,  12]	[6,  10]
% [6,  13]	[4,  8 ]	[2,  7 ]
% [3,  7 ]	[2,  12]	[9,  10]
% [8,  11]	[4,  10]	[5,  13]
% [4,  7 ]	[3,  5 ]	[2,  11]
% [9,  12]	[6,  7 ]	[3,  10]
% [2,  13]	[4,  11]	[8,  12]
% [3,  13]	[5,  10]	[4,  9 ]
% [8,  10]	[5,  9 ]	[3,  6 ]
% --------	--------	--------
% --------	--------	--------
% --------	--------	--------
% --------	--------	--------