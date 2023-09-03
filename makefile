
STD = c++17

OBJECT_FILES = \
	build/main.o\
	build/lexer.o\
	build/engine.o

all: quetza

quetza: $(OBJECT_FILES) | build
	$(CXX) $(OBJECT_FILES) -O3 -o quetza

build:
	@ if [ ! -d build ]; then mkdir build; fi

build/main.o: src/engine.h src/lexer.h src/main.cpp | build
	$(CXX) -std=$(STD) -c -O3 src/main.cpp -fno-exceptions -fPIC -o build/main.o -I .

build/lexer.o: src/lexer.h src/lexer.cpp | build
	$(CXX) -std=$(STD) -c -O3 src/lexer.cpp -fno-exceptions -fPIC -o build/lexer.o -I .

build/engine.o: src/engine.h src/engine.cpp | build
	$(CXX) -std=$(STD) -c -O3 src/engine.cpp -fno-exceptions -fPIC -o build/engine.o -I .

.PHONY = clean
clean: build
	@ rm build/*.o