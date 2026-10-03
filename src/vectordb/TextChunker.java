package vectordb;

import java.util.ArrayList;
import java.util.List;

/** Splits text into overlapping word windows for embedding. */
public final class TextChunker {
    private TextChunker() {}

    public static List<String> chunk(String text, int chunkWords, int overlapWords) {
        String trimmed = text.strip();
        List<String> out = new ArrayList<>();
        if (trimmed.isEmpty()) return out;

        String[] words = trimmed.split("\\s+");
        if (words.length <= chunkWords) {
            out.add(text); // short text stays untouched, like the C++ version
            return out;
        }
        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            out.add(String.join(" ", java.util.Arrays.copyOfRange(words, i, end)));
            if (end == words.length) break;
        }
        return out;
    }
}
