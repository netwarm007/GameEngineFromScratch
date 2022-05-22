/* recongnize tokens for the calculator and print them out */
%option noyywrap

%%
"+"		{ printf("加\n"); }
"-"		{ printf("减\n"); }
"*"		{ printf("乘\n"); }
"/"		{ printf("除\n"); }
"|"		{ printf("绝对值\n"); }
[0-9]+		{ printf("数 %s\n", yytext); }
\n		{ printf("换行\n"); }
[ \t]		{ }
.		{ printf("一些神奇的字符 %s\n", yytext); }
%%

int main()
{
    yylex();
    return 0;
}
