.PHONY: configure build test clean

BUILD_DIR ?= build/debug
BUILD_TYPE ?= Debug
UOCR_ENABLE_METAL ?= AUTO

configure:
	cmake -S . -B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DUOCR_ENABLE_METAL=$(UOCR_ENABLE_METAL) \
		-DUOCR_BUILD_TOOLS=ON \
		-DUOCR_BUILD_TESTS=ON

build: configure
	cmake --build $(BUILD_DIR) -j

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	rm -rf build
