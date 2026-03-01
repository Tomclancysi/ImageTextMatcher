(function () {
  const { createApp } = Vue;

  const config = window.__ITM_CONFIG__ || {};
  const HISTORY_KEY = "itm-search-history";
  const methods = [
    { value: "clip", label: "CLIP" },
    { value: "vse", label: "VSE++" },
    { value: "scan", label: "SCAN" },
  ];
  const quickExamples = [
    "red train at a station",
    "person riding a bicycle",
    "city street with cars",
    "snowy mountain landscape",
  ];

  function readHistory() {
    try {
      const raw = window.localStorage.getItem(HISTORY_KEY);
      if (!raw) {
        return [];
      }
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
      return [];
    }
  }

  function writeHistory(history) {
    try {
      window.localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 6)));
    } catch (error) {
      return;
    }
  }

  const app = createApp({
    data() {
      return {
        methods,
        quickExamples,
        form: {
          query: config.initialQuery || "",
          topK: Number(config.currentTopK || 10),
          method: config.currentMethod || "clip",
        },
        viewMode: "single",
        loading: false,
        error: "",
        results: [],
        compareResults: {},
        queryVector: [],
        showVectorPreview: true,
        showQueryVectorPanel: false,
        scoreFloor: 0,
        correction: {
          visible: false,
          original: "",
          corrected: "",
          suggestions: [],
        },
        responseMeta: {
          query: "",
          correctedQuery: "",
          durationMs: null,
        },
        details: {
          visible: false,
          item: null,
          sourceLabel: "",
        },
        history: readHistory(),
      };
    },
    computed: {
      compareMode() {
        return this.viewMode === "compare";
      },
      hasSearched() {
        return this.responseMeta.query.length > 0;
      },
      methodLabel() {
        return this.labelForMethod(this.form.method);
      },
      activeResults() {
        if (this.compareMode) {
          return Object.values(this.compareResults).flatMap((column) => column.results || []);
        }
        return this.results;
      },
      scoreStats() {
        if (!this.activeResults.length) {
          return { min: 0, max: 1 };
        }
        const scores = this.activeResults.map((item) => Number(item.score) || 0);
        return {
          min: Math.min.apply(null, scores),
          max: Math.max.apply(null, scores),
        };
      },
      displayedResults() {
        return this.results.filter((item) => Number(item.score) >= this.scoreFloor);
      },
      compareColumns() {
        return this.methods.map((method) => {
          const base = this.compareResults[method.value] || {};
          const results = Array.isArray(base.results) ? base.results : [];
          const filteredResults = results.filter((item) => Number(item.score) >= this.scoreFloor);
          return {
            method: method.value,
            label: method.label,
            durationMs: base.durationMs ?? null,
            error: base.error || "",
            filteredResults,
            visibleCount: filteredResults.length,
          };
        });
      },
      stats() {
        const totalResults = this.compareMode
          ? this.compareColumns.reduce((sum, column) => sum + column.visibleCount, 0)
          : this.displayedResults.length;
        const durationMs = this.compareMode
          ? this.methods.reduce((max, method) => {
              const column = this.compareResults[method.value];
              const value = column && typeof column.durationMs === "number" ? column.durationMs : null;
              return value !== null ? Math.max(max, value) : max;
            }, 0) || null
          : this.responseMeta.durationMs;
        return {
          query: this.responseMeta.query,
          correctedQuery: this.responseMeta.correctedQuery,
          totalResults: this.hasSearched ? totalResults : null,
          durationMs,
        };
      },
      skeletonCount() {
        return Math.min(Math.max(this.form.topK, 3), 8);
      },
    },
    mounted() {
      if (this.form.query) {
        this.submitSearch();
      }
    },
    methods: {
      async submitSearch() {
        const query = (this.form.query || "").trim();
        if (!query) {
          this.error = "Please enter a text description before searching.";
          return;
        }

        this.form.query = query;
        this.loading = true;
        this.error = "";
        this.details.visible = false;

        try {
          if (this.compareMode) {
            await this.runCompareSearch();
          } else {
            await this.runSingleSearch(this.form.method);
          }
          this.scoreFloor = this.scoreStats.min;
          this.pushHistory();
          this.syncUrl();
        } catch (error) {
          this.error = error.message || "Unexpected search error.";
        } finally {
          this.loading = false;
        }
      },
      async runSingleSearch(method) {
        const payload = await this.fetchSearch(method);
        this.results = Array.isArray(payload.results) ? payload.results : [];
        this.compareResults = {};
        this.queryVector = Array.isArray(payload.query_vector_summary) ? payload.query_vector_summary : [];
        this.responseMeta = {
          query: payload.query || this.form.query,
          correctedQuery: payload.corrected_query || this.form.query,
          durationMs: payload.duration_ms ?? null,
        };
        this.correction = {
          visible: Boolean(payload.corrected_query && payload.corrected_query !== payload.query),
          original: payload.query || this.form.query,
          corrected: payload.corrected_query || this.form.query,
          suggestions: Array.isArray(payload.suggestions) ? payload.suggestions : [],
        };
      },
      async runCompareSearch() {
        const responses = await Promise.all(
          this.methods.map(async (method) => {
            try {
              const payload = await this.fetchSearch(method.value);
              return {
                method: method.value,
                payload,
                error: "",
              };
            } catch (error) {
              return {
                method: method.value,
                payload: null,
                error: error.message || "Search failed.",
              };
            }
          })
        );

        this.results = [];
        this.compareResults = {};

        responses.forEach((entry, index) => {
          if (entry.payload) {
            this.compareResults[entry.method] = {
              results: Array.isArray(entry.payload.results) ? entry.payload.results : [],
              durationMs: entry.payload.duration_ms ?? null,
              error: "",
            };
            if (index === 0) {
              this.queryVector = Array.isArray(entry.payload.query_vector_summary) ? entry.payload.query_vector_summary : [];
              this.responseMeta = {
                query: entry.payload.query || this.form.query,
                correctedQuery: entry.payload.corrected_query || this.form.query,
                durationMs: entry.payload.duration_ms ?? null,
              };
              this.correction = {
                visible: Boolean(entry.payload.corrected_query && entry.payload.corrected_query !== entry.payload.query),
                original: entry.payload.query || this.form.query,
                corrected: entry.payload.corrected_query || this.form.query,
                suggestions: Array.isArray(entry.payload.suggestions) ? entry.payload.suggestions : [],
              };
            }
          } else {
            this.compareResults[entry.method] = {
              results: [],
              durationMs: null,
              error: entry.error,
            };
          }
        });

        const allFailed = responses.every((entry) => !entry.payload);
        if (allFailed) {
          throw new Error("All model searches failed.");
        }
      },
      async fetchSearch(method) {
        const params = new URLSearchParams({
          q: this.form.query,
          k: String(this.form.topK),
          method,
        });
        const response = await fetch(`/api/search?${params.toString()}`, {
          headers: { Accept: "application/json" },
        });
        if (!response.ok) {
          throw new Error(`Search failed with status ${response.status}.`);
        }
        return response.json();
      },
      resetSearch() {
        this.form.query = "";
        this.form.topK = Number(config.currentTopK || 10);
        this.form.method = config.currentMethod || "clip";
        this.viewMode = "single";
        this.results = [];
        this.compareResults = {};
        this.queryVector = [];
        this.error = "";
        this.scoreFloor = 0;
        this.showVectorPreview = true;
        this.showQueryVectorPanel = false;
        this.correction = {
          visible: false,
          original: "",
          corrected: "",
          suggestions: [],
        };
        this.responseMeta = {
          query: "",
          correctedQuery: "",
          durationMs: null,
        };
        this.details.visible = false;
        this.syncUrl(true);
      },
      applyExample(example) {
        this.form.query = example;
        this.submitSearch();
      },
      applyHistory(item) {
        this.form.query = item.query;
        this.form.topK = item.topK;
        this.form.method = item.method;
        this.viewMode = item.viewMode || "single";
        this.submitSearch();
      },
      pushHistory() {
        const entry = {
          query: this.form.query,
          topK: this.form.topK,
          method: this.form.method,
          methodLabel: this.labelForMethod(this.form.method),
          viewMode: this.viewMode,
        };
        const next = [entry].concat(
          this.history.filter((item) => {
            return !(
              item.query === entry.query &&
              item.topK === entry.topK &&
              item.method === entry.method &&
              item.viewMode === entry.viewMode
            );
          })
        );
        this.history = next.slice(0, 6);
        writeHistory(this.history);
      },
      openDetails(item, sourceLabel) {
        this.details.item = item;
        this.details.sourceLabel = sourceLabel;
        this.details.visible = true;
      },
      historyKey(item) {
        return `${item.query}-${item.method}-${item.topK}-${item.viewMode || "single"}`;
      },
      summaryText(item) {
        const description = item && item.description;
        const text = description && description.global_caption
          ? description.global_caption
          : "No caption metadata is available for this image.";
        return text.length > 110 ? `${text.slice(0, 107)}...` : text;
      },
      basename(path) {
        if (!path) {
          return "";
        }
        const normalized = String(path).replace(/\\/g, "/");
        const parts = normalized.split("/");
        return parts[parts.length - 1];
      },
      labelForMethod(method) {
        const match = this.methods.find((item) => item.value === method);
        return match ? match.label : String(method || "").toUpperCase();
      },
      vectorColor(value) {
        const numeric = Number(value);
        const clamped = Number.isFinite(numeric) ? Math.min(1, Math.max(0, numeric)) : 0;
        const hue = Math.round((1 - clamped) * 220);
        const saturation = 78;
        const lightness = Math.round(36 + clamped * 22);
        return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
      },
      syncUrl(reset) {
        const url = new URL(window.location.href);
        if (reset) {
          url.search = "";
        } else {
          url.searchParams.set("q", this.form.query);
          url.searchParams.set("k", String(this.form.topK));
          url.searchParams.set("method", this.form.method);
        }
        window.history.replaceState({}, "", url.toString());
      },
    },
  });

  app.config.compilerOptions.delimiters = ["[[", "]]"];
  app.use(ElementPlus);
  app.mount("#app");
})();
