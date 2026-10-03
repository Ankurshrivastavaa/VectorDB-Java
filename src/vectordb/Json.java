package vectordb;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal, dependency-free JSON parser plus string-quoting helper.
 * parse() returns Map / List / String / Double / Boolean / null.
 *
 * Unlike the hand-rolled extractStr() in the C++ code, this handles
 * \\uXXXX escapes, nesting, and escaped quotes properly. Ollama emits
 * \\u003c style escapes, so that matters.
 */
public final class Json {
    private final String s;
    private int p = 0;

    private Json(String s) { this.s = s; }

    public static Object parse(String text) {
        Json j = new Json(text);
        j.ws();
        Object v = j.value();
        j.ws();
        if (j.p != j.s.length()) throw j.err("unexpected trailing data");
        return v;
    }

    // ---------- accessors for request bodies (lenient, return defaults) ----------

    @SuppressWarnings("unchecked")
    public static Map<String, Object> parseObject(String text) {
        try {
            Object o = parse(text);
            return o instanceof Map ? (Map<String, Object>) o : new LinkedHashMap<>();
        } catch (RuntimeException e) {
            return new LinkedHashMap<>();
        }
    }

    public static String getString(Map<String, Object> m, String key) {
        return m.get(key) instanceof String str ? str : "";
    }

    public static int getInt(Map<String, Object> m, String key, int def) {
        return m.get(key) instanceof Number n ? n.intValue() : def;
    }

    public static float[] getFloats(Map<String, Object> m, String key) {
        if (!(m.get(key) instanceof List<?> list)) return new float[0];
        float[] out = new float[list.size()];
        for (int i = 0; i < out.length; i++) {
            if (!(list.get(i) instanceof Number n)) return new float[0];
            out[i] = n.floatValue();
        }
        return out;
    }

    // ---------- writer helper ----------

    public static String quote(String str) {
        StringBuilder o = new StringBuilder(str.length() + 2).append('"');
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            switch (c) {
                case '"' -> o.append("\\\"");
                case '\\' -> o.append("\\\\");
                case '\n' -> o.append("\\n");
                case '\r' -> o.append("\\r");
                case '\t' -> o.append("\\t");
                default -> {
                    if (c < 0x20) o.append(String.format("\\u%04x", (int) c));
                    else o.append(c);
                }
            }
        }
        return o.append('"').toString();
    }

    // ---------- parser ----------

    private Object value() {
        if (p >= s.length()) throw err("unexpected end");
        char c = s.charAt(p);
        switch (c) {
            case '{': return object();
            case '[': return array();
            case '"': return string();
            case 't': lit("true"); return Boolean.TRUE;
            case 'f': lit("false"); return Boolean.FALSE;
            case 'n': lit("null"); return null;
            default: return number();
        }
    }

    private Map<String, Object> object() {
        Map<String, Object> m = new LinkedHashMap<>();
        p++; // {
        ws();
        if (peek() == '}') { p++; return m; }
        while (true) {
            ws();
            if (peek() != '"') throw err("expected string key");
            String key = string();
            ws();
            expect(':');
            ws();
            m.put(key, value());
            ws();
            if (peek() == ',') { p++; continue; }
            expect('}');
            return m;
        }
    }

    private List<Object> array() {
        List<Object> l = new ArrayList<>();
        p++; // [
        ws();
        if (peek() == ']') { p++; return l; }
        while (true) {
            ws();
            l.add(value());
            ws();
            if (peek() == ',') { p++; continue; }
            expect(']');
            return l;
        }
    }

    private String string() {
        StringBuilder sb = new StringBuilder();
        p++; // opening quote
        while (true) {
            if (p >= s.length()) throw err("unterminated string");
            char c = s.charAt(p++);
            if (c == '"') return sb.toString();
            if (c != '\\') { sb.append(c); continue; }
            if (p >= s.length()) throw err("bad escape");
            char e = s.charAt(p++);
            switch (e) {
                case '"', '\\', '/' -> sb.append(e);
                case 'b' -> sb.append('\b');
                case 'f' -> sb.append('\f');
                case 'n' -> sb.append('\n');
                case 'r' -> sb.append('\r');
                case 't' -> sb.append('\t');
                case 'u' -> {
                    if (p + 4 > s.length()) throw err("bad unicode escape");
                    sb.append((char) Integer.parseInt(s.substring(p, p + 4), 16));
                    p += 4;
                }
                default -> throw err("bad escape \\" + e);
            }
        }
    }

    private Double number() {
        int start = p;
        while (p < s.length() && "+-0123456789.eE".indexOf(s.charAt(p)) >= 0) p++;
        if (start == p) throw err("unexpected character '" + s.charAt(p) + "'");
        return Double.parseDouble(s.substring(start, p));
    }

    private void lit(String word) {
        if (!s.startsWith(word, p)) throw err("expected " + word);
        p += word.length();
    }

    private void expect(char c) {
        if (peek() != c) throw err("expected '" + c + "'");
        p++;
    }

    private char peek() { return p < s.length() ? s.charAt(p) : '\0'; }

    private void ws() {
        while (p < s.length() && Character.isWhitespace(s.charAt(p))) p++;
    }

    private IllegalArgumentException err(String msg) {
        return new IllegalArgumentException("JSON error at " + p + ": " + msg);
    }
}
