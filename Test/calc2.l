/* recognize tokens for the calculator and print them out */
%{
#include "calc.tab.h"
%}

%option noyywrap

%%
"+"     { return ADD; }
"-"     { return SUB; }
"*"     { return MUL; }
"/"     { return DIV; }
"|"     { return ABS; }
"("     { return OP;  }
")"     { return CP;  }
"//".*  { /* ignore comments */ }
"/*"([^*]|(\*+[^*/]))*\*+\/             { /* ignore C-style comments */ }
[0-9]+  { yylval = atoi(yytext); return NUMBER; }
\n      { return EOL; }
[ \t]   { /* ignore whitespace */ }
.       { printf("魔法字符 %c\n", *yytext); }
%%
