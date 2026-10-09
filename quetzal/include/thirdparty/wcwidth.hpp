#pragma once

int mk_wcwidth(char32_t ucs);
int mk_wcswidth(const char32_t *pwcs, size_t n);
int mk_wcwidth_cjk(char32_t ucs);
int mk_wcswidth_cjk(const char32_t *pwcs, size_t n);
