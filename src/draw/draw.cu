#include "draw.cuh"

void swap(float* a, float* b) {
    float tmp = *b;
    *b = *a;
    *a = tmp;
};

void drawCercle(Pixel* image, Point c, int radius, Pixel color) {
    int initialI = max((int)c.x - radius, 0);
    int finalI = min((int)c.x + radius, WINDOW_WIDTH);

    int initialJ = max((int)c.y - radius, 0);
    int finalJ = min((int)c.y + radius, WINDOW_HEIGHT);

    int radiusSq = radius * radius;

    for (int i = initialI; i < finalI; i++) {
        for (int j = initialJ; j < finalJ; j++) {
            if (powf(i - c.x, 2) + powf(j - c.y, 2) > radiusSq) continue;
            image[i + (j)*WINDOW_WIDTH] = color;
        }
    }
}

void drawSquare(Pixel* image, Point c, int width, Pixel color) {
    int initialI = max((int)c.x - width, 0);
    int finalI = min((int)c.x + width, WINDOW_WIDTH);

    int initialJ = max((int)c.y - width / 2, 0);
    int finalJ = min((int)c.y + width / 2, WINDOW_HEIGHT);

    for (int i = initialI; i < finalI; i++) {
        for (int j = initialJ; j < finalJ; j++) {
            image[i + (j)*WINDOW_WIDTH] = color;
        }
    }
}

void drawLine(Pixel* image, Point s, Point d, Pixel color) {
    float x1 = s.x;
    float y1 = s.y;

    float x2 = d.x;
    float y2 = d.y;

    float rise = y2 - y1;
    float run = x2 - x1;

    if (run == 0) {
        if (y2 < y1) {
            swap(&y1, &y2);
        }

        for (int y = y1; y < y2 + 1; y++) {
            int pos = (int)x1 + y * WINDOW_HEIGHT;
            if (pos < WINDOW_HEIGHT * WINDOW_WIDTH && pos > 0) image[pos] = color;
        }
    } else {
        float m = (float)rise / run;
        int adjust = m >= 0 ? 1 : -1;
        float offset = 0;
        float threshold = 0.5;

        if (m <= 1 && m >= -1) {
            float delta = fabs(m);

            int y = y1;
            if (x2 < x1) {
                swap(&x1, &x2);
                y = y2;
            }

            for (int x = x1; x < x2 + 1; x++) {
                int pos = x + y * WINDOW_HEIGHT;
                if (pos < WINDOW_HEIGHT * WINDOW_WIDTH && pos > 0) image[pos] = color;
                offset += delta;
                if (offset >= threshold) {
                    y += adjust;
                    threshold += 1;
                }
            }
        } else {
            float delta = fabs((float)run / rise);
            int x = x1;
            if (y2 < y1) {
                swap(&y1, &y2);
                x = x2;
            }

            for (int y = y1; y < y2 + 1; y++) {
                int pos = x + y * WINDOW_HEIGHT;
                if (pos < WINDOW_HEIGHT * WINDOW_WIDTH && pos > 0) image[pos] = color;
                offset += delta;
                if (offset >= threshold) {
                    x += adjust;
                    threshold += 1;
                }
            }
        }
    }
}

__device__ void cudaDrawCercle(Pixel* image, Point c, int radius, Pixel color) {
    int initialI = max((int)c.x - radius, 0);
    int finalI = min((int)c.x + radius, WINDOW_WIDTH);

    int initialJ = max((int)c.y - radius, 0);
    int finalJ = min((int)c.y + radius, WINDOW_HEIGHT);

    int radiusSq = radius * radius;

    for (int i = initialI; i < finalI; i++) {
        for (int j = initialJ; j < finalJ; j++) {
            if (powf(i - c.x, 2) + powf(j - c.y, 2) > radiusSq) continue;
            image[i + (j)*WINDOW_WIDTH] = color;
        }
    }
}

__device__ void cudaDrawSquare(Pixel* image, Point c, int width, Pixel color) {
    int initialI = max((int)c.x - width, 0);
    int finalI = min((int)c.x + width, WINDOW_WIDTH);

    int initialJ = max((int)c.y - width, 0);
    int finalJ = min((int)c.y + width, WINDOW_HEIGHT);

    for (int i = initialI; i < finalI; i++) {
        for (int j = initialJ; j < finalJ; j++) {
            image[i + (j)*WINDOW_WIDTH] = color;
        }
    }
}

__device__ void cudaSwap(float& y1, float& y2) {
    float tmp = y1;
    y1 = y2;
    y2 = tmp;
}

__device__ void cudaDrawLine(Pixel* image, Point s, Point d, Pixel color) {
    float x1 = s.x;
    float y1 = s.y;

    float x2 = d.x;
    float y2 = d.y;

    float rise = y2 - y1;
    float run = x2 - x1;

    if (run == 0) {
        if (y2 < y1) {
            cudaSwap(y1, y2);
        }

        for (int y = y1; y < y2 + 1; y++) {
            int pos = (int)x1 + y * WINDOW_HEIGHT;
            if (pos < WINDOW_HEIGHT * WINDOW_WIDTH && pos > 0) image[pos] = color;
        }
    } else {
        float m = (float)rise / run;
        int adjust = m >= 0 ? 1 : -1;
        float offset = 0;
        float threshold = 0.5;

        if (m <= 1 and m >= -1) {
            float delta = fabs(m);

            int y = y1;
            if (x2 < x1) {
                cudaSwap(x1, x2);
                y = y2;
            }

            for (int x = x1; x < x2 + 1; x++) {
                int pos = x + y * WINDOW_HEIGHT;
                if (pos < WINDOW_HEIGHT * WINDOW_WIDTH && pos > 0) image[pos] = color;
                offset += delta;
                if (offset >= threshold) {
                    y += adjust;
                    threshold += 1;
                }
            }
        } else {
            float delta = fabs((float)run / rise);
            int x = x1;
            if (y2 < y1) {
                cudaSwap(y1, y2);
                x = x2;
            }

            for (int y = y1; y < y2 + 1; y++) {
                int pos = x + y * WINDOW_HEIGHT;
                if (pos < WINDOW_HEIGHT * WINDOW_WIDTH && pos > 0) image[pos] = color;
                offset += delta;
                if (offset >= threshold) {
                    x += adjust;
                    threshold += 1;
                }
            }
        }
    }
}
