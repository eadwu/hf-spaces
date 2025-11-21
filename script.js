import * as ort from 'onnxruntime-web';
const presetTexts = window.presetTexts || {};

const PLAY_ICON_SVG = `<svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" focusable="false"><path d="M8 5v14l11-7-11-7z"></path></svg>`;
const PAUSE_ICON_SVG = `<svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" focusable="false"><path d="M8 6h3v12H8V6zm5 0h3v12h-3V6z"></path></svg>`;
const STOP_ICON_SVG = `<svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" focusable="false"><path d="M7 7h10v10H7V7z"></path></svg>`;

// Lightning background parallax
(function initLightningParallax() {
    if (typeof document === 'undefined') {
        return;
    }

    // Removed scroll-based CSS variable updates for direct scroll response
    // const updateLightningOffset = () => {
    //     document.body.style.setProperty('--lightning-scroll', `${window.scrollY}px`);
    // };
    // let ticking = false;
    // const onScroll = () => {
    //     if (!ticking) {
    //         window.requestAnimationFrame(() => {
    //             updateLightningOffset();
    //             ticking = false;
    //         });
    //         ticking = true;
    //     }
    // };
    // updateLightningOffset();
    // window.addEventListener('scroll', onScroll, { passive: true });

    const runBlink = (className, onComplete) => {
        let remaining = 1 + Math.round(Math.random());
        const blink = () => {
            if (remaining-- <= 0) {
                if (typeof onComplete === 'function') {
                    onComplete();
                }
                return;
            }
            const wait = 20 + Math.random() * 80;
            document.body.classList.add(className);
            setTimeout(() => {
                document.body.classList.remove(className);
                setTimeout(blink, wait);
            }, wait);
        };
        blink();
    };

    const schedule = () => {
        setTimeout(() => runBlink('lightning-flicker', schedule), Math.random() * 10000);
    };
    schedule();

    /*
    const heroSection = document.querySelector('.hero');
    if (heroSection) {
        heroSection.addEventListener('click', (event) => {
            runBlink('lightning-flash');
        });
    }
    */
})();

function escapeHtml(value) {
    return value.replace(/[&<>"']/g, (match) => {
        switch (match) {
            case '&': return '&amp;';
            case '<': return '&lt;';
            case '>': return '&gt;';
            case '"': return '&quot;';
            case "'": return '&#39;';
            default: return match;
        }
    });
}

function formatStatValueWithSuffix(value, suffix, options = {}) {
    const { firstLabel = false } = options;
    if (value === undefined || value === null) {
        return '';
    }
    if (!suffix) {
        const raw = `${value}`;
        return escapeHtml(raw);
    }
    const raw = `${value}`.trim();
    if (!raw || raw === '--' || raw === '-' || raw.toLowerCase() === 'error') {
        return escapeHtml(raw);
    }
    const appendSuffix = (segment, includePrefix = false) => {
        const trimmed = segment.trim();
        if (!trimmed) {
            return '';
        }
        const escapedValue = `<span class="stat-value-number">${escapeHtml(trimmed)}</span>`;
        const suffixSpan = `<span class="stat-label stat-suffix">${escapeHtml(suffix)}</span>`;
        const prefixSpan = includePrefix && firstLabel
            ? `<span class="stat-label stat-suffix stat-prefix">First</span>`
            : '';
        const segmentClass = includePrefix && firstLabel
            ? 'stat-value-segment has-prefix'
            : 'stat-value-segment';
        return `<span class="${segmentClass}">${prefixSpan}${escapedValue}${suffixSpan}</span>`;
    };
    if (raw.includes('/')) {
        const parts = raw.split('/');
        const segments = parts.map((part, index) => appendSuffix(part, index === 0));
        return segments.join(' / ');
    }
    return appendSuffix(raw);
}

/**
 * Unicode text processor
 */
export class UnicodeProcessor {
    constructor(indexer) {
        this.indexer = indexer;
    }

    call(textList) {
        const processedTexts = textList.map(t => preprocessText(t));
        const textIdsLengths = processedTexts.map(t => t.length);
        const maxLen = Math.max(...textIdsLengths);
        
        const textIds = [];
        const unsupportedChars = new Set();
        
        for (let i = 0; i < processedTexts.length; i++) {
            const row = new Array(maxLen).fill(0);
            const unicodeVals = textToUnicodeValues(processedTexts[i]);
            for (let j = 0; j < unicodeVals.length; j++) {
                const indexValue = this.indexer[unicodeVals[j]];
                // Check if character is supported (not -1, undefined, or null)
                if (indexValue === undefined || indexValue === null || indexValue === -1) {
                    unsupportedChars.add(processedTexts[i][j]);
                    row[j] = 0; // Use 0 as fallback
                } else {
                    row[j] = indexValue;
                }
            }
            textIds.push(row);
        }
        
        const textMask = getTextMask(textIdsLengths);
        return { textIds, textMask, unsupportedChars: Array.from(unsupportedChars) };
    }
}

export function preprocessText(text) {
    // Normalize unicode characters
    text = text.normalize('NFKD');
    
    // Remove emojis
    text = text.replace(/[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F700}-\u{1F77F}\u{1F780}-\u{1F7FF}\u{1F800}-\u{1F8FF}\u{1F900}-\u{1F9FF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{1F1E6}-\u{1F1FF}]/gu, '');
    
    // Replace various dashes and symbols
    text = text.replace(/–/g, "-");
    text = text.replace(/‑/g, "-");
    text = text.replace(/—/g, "-");
    text = text.replace(/¯/g, " ");
    text = text.replace(/_/g, " ");
    text = text.replace(/[“”]/g, '"');
    text = text.replace(/[‘’´`]/g, "'");
    text = text.replace(/\[/g, " ");
    text = text.replace(/\]/g, " ");
    text = text.replace(/\|/g, " ");
    text = text.replace(/[\u0302\u0303\u0304\u0305\u0306\u0307\u0308\u030A\u030B\u030C\u0327\u0328\u0329\u032A\u032B\u032C\u032D\u032E\u032F]/g, "");
    text = text.replace(/\//g, " "); // FIXME: slash should be kept (e.g., fraction)
    text = text.replace(/#/g, " "); // FIXME: hash should be kept (e.g., hashtag)
    text = text.replace(/→/g, " ");
    text = text.replace(/←/g, " ");


    // Remove special symbols
    text = text.replace(/[♥☆♡©\\]/g, "");

    // Replace known expressions
    text = text.replace(/@/g, " at ");
    text = text.replace(/&/g, " and ");
    text = text.replace(/e\.g\.,/g, "for example, ");
    text = text.replace(/i\.e\.,/g, "that is, ");

    
    // Fix spacing around punctuation
    text = text.replace(/ ,/g, ",");
    text = text.replace(/ \./g, ".");
    text = text.replace(/ !/g, "!");
    text = text.replace(/ \?/g, "?");
    text = text.replace(/ ;/g, ";");
    text = text.replace(/ :/g, ":");
    text = text.replace(/ '/g, "'");
    
    // Remove duplicate quotes
    while (text.includes('""')) {
        text = text.replace(/""/g, '"');
    }
    while (text.includes("''")) {
        text = text.replace(/''/g, "'");
    }
    while (text.includes("``")) {
        text = text.replace(/``/g, "`");
    }
    
    // Remove extra spaces
    while (text.includes("  ")) {
        text = text.replace(/  /g, " ");
    }
    
    // Remove first and last spaces
    text = text.trim();
    text = text.replace(/\s+/g, " ");  // Remove extra spaces

    // if text doesn't end with punctuation, quotes, or closing brackets, add a period
    const lastChar = text[text.length - 1];
    if (!/[.!?;:,'"'"')\]}…。」』】〉》›»]/.test(lastChar)) {
        text = text + '.';
    }
    
    return text;
}

export function textToUnicodeValues(text) {
    return Array.from(text).map(char => char.charCodeAt(0));
}

export function lengthToMask(lengths, maxLen = null) {
    maxLen = maxLen || Math.max(...lengths);
    const mask = [];
    for (let i = 0; i < lengths.length; i++) {
        const row = [];
        for (let j = 0; j < maxLen; j++) {
            row.push(j < lengths[i] ? 1.0 : 0.0);
        }
        mask.push([row]);
    }
    return mask;
}

export function getTextMask(textIdsLengths) {
    return lengthToMask(textIdsLengths);
}

export function getLatentMask(wavLengths, cfgs) {
    const baseChunkSize = cfgs.ae.base_chunk_size;
    const chunkCompressFactor = cfgs.ttl.chunk_compress_factor;
    const latentSize = baseChunkSize * chunkCompressFactor;
    const latentLengths = wavLengths.map(len => 
        Math.floor((len + latentSize - 1) / latentSize)
    );
    return lengthToMask(latentLengths);
}

export function sampleNoisyLatent(duration, cfgs) {
    const sampleRate = cfgs.ae.sample_rate;
    const baseChunkSize = cfgs.ae.base_chunk_size;
    const chunkCompressFactor = cfgs.ttl.chunk_compress_factor;
    const ldim = cfgs.ttl.latent_dim;

    const wavLenMax = Math.max(...duration.map(d => d[0][0])) * sampleRate;
    const wavLengths = duration.map(d => Math.floor(d[0][0] * sampleRate));
    const chunkSize = baseChunkSize * chunkCompressFactor;
    const latentLen = Math.floor((wavLenMax + chunkSize - 1) / chunkSize);
    const latentDim = ldim * chunkCompressFactor;

    const noisyLatent = [];
    for (let b = 0; b < duration.length; b++) {
        const batch = [];
        for (let d = 0; d < latentDim; d++) {
            const row = [];
            for (let t = 0; t < latentLen; t++) {
                const u1 = Math.random();
                const u2 = Math.random();
                const randNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
                row.push(randNormal);
            }
            batch.push(row);
        }
        noisyLatent.push(batch);
    }

    const latentMask = getLatentMask(wavLengths, cfgs);
    
    for (let b = 0; b < noisyLatent.length; b++) {
        for (let d = 0; d < noisyLatent[b].length; d++) {
            for (let t = 0; t < noisyLatent[b][d].length; t++) {
                noisyLatent[b][d][t] *= latentMask[b][0][t];
            }
        }
    }

    return { noisyLatent, latentMask };
}

export async function loadOnnx(onnxPath, opts) {
    return await ort.InferenceSession.create(onnxPath, opts);
}

export async function loadOnnxAll(basePath, opts, onProgress) {
    const models = [
        { name: 'Duration Predictor', path: `${basePath}/duration_predictor.onnx`, key: 'dpOrt' },
        { name: 'Text Encoder', path: `${basePath}/text_encoder.onnx`, key: 'textEncOrt' },
        { name: 'Vector Estimator', path: `${basePath}/vector_estimator.onnx`, key: 'vectorEstOrt' },
        { name: 'Vocoder', path: `${basePath}/vocoder.onnx`, key: 'vocoderOrt' }
    ];

    const result = {};
    let loadedCount = 0;
    
    // Load all models in parallel
    const loadPromises = models.map(async (model) => {
        const session = await loadOnnx(model.path, opts);
        loadedCount++;
        if (onProgress) {
            onProgress(model.name, loadedCount, models.length);
        }
        return { key: model.key, session };
    });
    
    // Wait for all models to load
    const loadedModels = await Promise.all(loadPromises);
    
    // Organize results
    loadedModels.forEach(({ key, session }) => {
        result[key] = session;
    });

    return result;
}

export async function loadCfgs(basePath) {
    const response = await fetch(`${basePath}/tts.json`);
    return await response.json();
}

export async function loadProcessors(basePath) {
    const response = await fetch(`${basePath}/unicode_indexer.json`);
    const unicodeIndexerData = await response.json();
    const textProcessor = new UnicodeProcessor(unicodeIndexerData);
    
    return { textProcessor };
}

function parseWavFile(buffer) {
    const view = new DataView(buffer);
    
    // Check RIFF header
    const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
    if (riff !== 'RIFF') {
        throw new Error('Not a valid WAV file');
    }
    
    const wave = String.fromCharCode(view.getUint8(8), view.getUint8(9), view.getUint8(10), view.getUint8(11));
    if (wave !== 'WAVE') {
        throw new Error('Not a valid WAV file');
    }
    
    let offset = 12;
    let fmtChunk = null;
    let dataChunk = null;
    
    while (offset < buffer.byteLength) {
        const chunkId = String.fromCharCode(
            view.getUint8(offset), 
            view.getUint8(offset + 1), 
            view.getUint8(offset + 2), 
            view.getUint8(offset + 3)
        );
        const chunkSize = view.getUint32(offset + 4, true);
        
        if (chunkId === 'fmt ') {
            fmtChunk = {
                audioFormat: view.getUint16(offset + 8, true),
                numChannels: view.getUint16(offset + 10, true),
                sampleRate: view.getUint32(offset + 12, true),
                bitsPerSample: view.getUint16(offset + 22, true)
            };
        } else if (chunkId === 'data') {
            dataChunk = {
                offset: offset + 8,
                size: chunkSize
            };
            break;
        }
        
        offset += 8 + chunkSize;
    }
    
    if (!fmtChunk || !dataChunk) {
        throw new Error('Invalid WAV file format');
    }
    
    const bytesPerSample = fmtChunk.bitsPerSample / 8;
    const numSamples = Math.floor(dataChunk.size / (bytesPerSample * fmtChunk.numChannels));
    const audioData = new Float32Array(numSamples);
    
    if (fmtChunk.bitsPerSample === 16) {
        for (let i = 0; i < numSamples; i++) {
            let sample = 0;
            for (let ch = 0; ch < fmtChunk.numChannels; ch++) {
                const sampleOffset = dataChunk.offset + (i * fmtChunk.numChannels + ch) * 2;
                sample += view.getInt16(sampleOffset, true);
            }
            audioData[i] = (sample / fmtChunk.numChannels) / 32768.0;
        }
    } else if (fmtChunk.bitsPerSample === 24) {
        // Support 24-bit PCM
        for (let i = 0; i < numSamples; i++) {
            let sample = 0;
            for (let ch = 0; ch < fmtChunk.numChannels; ch++) {
                const sampleOffset = dataChunk.offset + (i * fmtChunk.numChannels + ch) * 3;
                // Read 3 bytes and convert to signed 24-bit integer
                const byte1 = view.getUint8(sampleOffset);
                const byte2 = view.getUint8(sampleOffset + 1);
                const byte3 = view.getUint8(sampleOffset + 2);
                let value = (byte3 << 16) | (byte2 << 8) | byte1;
                // Convert to signed (two's complement)
                if (value & 0x800000) {
                    value = value - 0x1000000;
                }
                sample += value;
            }
            audioData[i] = (sample / fmtChunk.numChannels) / 8388608.0; // 2^23
        }
    } else if (fmtChunk.bitsPerSample === 32) {
        for (let i = 0; i < numSamples; i++) {
            let sample = 0;
            for (let ch = 0; ch < fmtChunk.numChannels; ch++) {
                const sampleOffset = dataChunk.offset + (i * fmtChunk.numChannels + ch) * 4;
                sample += view.getFloat32(sampleOffset, true);
            }
            audioData[i] = sample / fmtChunk.numChannels;
        }
    } else {
        throw new Error(`Unsupported bit depth: ${fmtChunk.bitsPerSample}. Supported formats: 16-bit, 24-bit, 32-bit`);
    }
    
    return {
        sampleRate: fmtChunk.sampleRate,
        audioData: audioData
    };
}

export function arrayToTensor(array, dims) {
    const flat = array.flat(Infinity);
    return new ort.Tensor('float32', Float32Array.from(flat), dims);
}

export function intArrayToTensor(array, dims) {
    const flat = array.flat(Infinity);
    return new ort.Tensor('int64', BigInt64Array.from(flat.map(x => BigInt(x))), dims);
}

export function writeWavFile(audioData, sampleRate) {
    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = sampleRate * numChannels * bitsPerSample / 8;
    const blockAlign = numChannels * bitsPerSample / 8;
    const dataSize = audioData.length * bitsPerSample / 8;

    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);
    
    // RIFF header
    view.setUint8(0, 'R'.charCodeAt(0));
    view.setUint8(1, 'I'.charCodeAt(0));
    view.setUint8(2, 'F'.charCodeAt(0));
    view.setUint8(3, 'F'.charCodeAt(0));
    view.setUint32(4, 36 + dataSize, true);
    view.setUint8(8, 'W'.charCodeAt(0));
    view.setUint8(9, 'A'.charCodeAt(0));
    view.setUint8(10, 'V'.charCodeAt(0));
    view.setUint8(11, 'E'.charCodeAt(0));
    
    // fmt chunk
    view.setUint8(12, 'f'.charCodeAt(0));
    view.setUint8(13, 'm'.charCodeAt(0));
    view.setUint8(14, 't'.charCodeAt(0));
    view.setUint8(15, ' '.charCodeAt(0));
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);
    
    // data chunk
    view.setUint8(36, 'd'.charCodeAt(0));
    view.setUint8(37, 'a'.charCodeAt(0));
    view.setUint8(38, 't'.charCodeAt(0));
    view.setUint8(39, 'a'.charCodeAt(0));
    view.setUint32(40, dataSize, true);
    
    // Write audio data
    for (let i = 0; i < audioData.length; i++) {
        const sample = Math.max(-1, Math.min(1, audioData[i]));
        const intSample = Math.floor(sample * 32767);
        view.setInt16(44 + i * 2, intSample, true);
    }
    
    return buffer;
}



// Smooth scroll functionality
document.addEventListener('DOMContentLoaded', () => {
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add scroll animation for sections
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe language icons and paper cards
    document.querySelectorAll('.language-icon, .paper-card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
        observer.observe(card);
    });
    
    // Add parallax effect to hero background
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const heroBg = document.querySelector('.hero-bg');
        if (heroBg) {
            heroBg.style.transform = `translateY(${scrolled * 0.5}px)`;
        }
    });

    const paperCards = document.querySelectorAll('.paper-card[data-link]');
    paperCards.forEach((card) => {
        const href = card.dataset.link ? card.dataset.link.trim() : '';
        if (!href) {
            return;
        }

        const openLink = () => {
            window.open(href, '_blank', 'noopener,noreferrer');
        };

        card.addEventListener('click', (event) => {
            if (event.defaultPrevented) {
                return;
            }
            openLink();
        });

        card.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                openLink();
            }
        });
    });
    
    // Active side navigation dot on scroll
    const sections = document.querySelectorAll('section[id]');
    const navDots = document.querySelectorAll('.nav-dot');
    
    const languageSnippetElement = document.querySelector('.languages-placeholder [data-language-snippet]');
    const languageTabButtons = Array.from(document.querySelectorAll('.languages-placeholder .language-option'));
    const languageIconButtons = Array.from(document.querySelectorAll('.languages-icons-container .language-icon'));
    const languageCopyBtn = document.querySelector('.languages-placeholder .code-copy-btn');
    const languageCopyToast = document.querySelector('.languages-placeholder .code-copy-toast');
    let copyToastTimeout = null;
    
    if (languageSnippetElement && (languageTabButtons.length || languageIconButtons.length)) {
        const sharedSetupSteps = [
            { text: '# Clone the Supertonic repository', type: 'heading' },
            { text: 'git clone https://github.com/supertone-inc/supertonic.git', type: 'command' },
            { text: 'cd supertonic', type: 'command' },
            { text: ' ', type: 'plain' },
            { text: '# Download ONNX models (NOTE: Make sure git-lfs is installed)', type: 'heading' },
            { text: 'git clone https://huggingface.co/Supertone/supertonic assets', type: 'command' },
            { text: ' ', type: 'plain' },
        ];

        const perLanguageCommands = {
            python: [
                'cd py',
                'uv sync',
                'uv run example_onnx.py',
            ],
            javascript: [
                'cd nodejs',
                'npm install',
                'npm start',
            ],
            java: [
                'cd java',
                'mvn clean install',
                'mvn exec:java',
            ],
            cpp: [
                'cd cpp',
                'mkdir build && cd build',
                'cmake .. && cmake --build . --config Release',
                './example_onnx',
            ],
            csharp: [
                'cd csharp',
                'dotnet restore',
                'dotnet run',
            ],
            go: [
                'cd go',
                'go mod download',
                'go run example_onnx.go helper.go',
            ],
            swift: [
                'cd swift',
                'swift build -c release',
                '.build/release/example_onnx',
            ],
            rust: [
                'cd rust',
                'cargo build --release',
                './target/release/example_onnx',
            ],
        };

        const buildCodeSample = (commands = []) => [
            ...sharedSetupSteps,
            { text: '# Run example', type: 'heading' },
            ...commands.map(text => ({ text, type: 'command' })),
        ];

        const codeSamples = Object.fromEntries(
            Object.entries(perLanguageCommands).map(([language, commands]) => [
                language,
                buildCodeSample(commands),
            ]),
        );
        
        const escapeHtml = (value) => value
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
        
        const renderSnippet = (lines) => lines.map((line) => {
            if (!line || typeof line.text !== 'string') {
                return '';
            }

            const kind = line.type || (line.highlight ? 'command' : 'plain');

            if (kind === 'heading') {
                return `<span class="token-heading">${escapeHtml(line.text)}</span>`;
            }

            if (kind === 'command') {
                const [command, ...rest] = line.text.trim().split(/\s+/);
                if (!command) {
                    return '';
                }
                const commandHtml = `<span class="token-command">${escapeHtml(command)}</span>`;
                const restHtml = rest.length
                    ? ` <span class="token-argument">${escapeHtml(rest.join(' '))}</span>`
                    : '';
                return `${commandHtml}${restHtml}`;
            }

            return `<span class="token-argument">${escapeHtml(line.text)}</span>`;
        }).join('\n');
        
        const setLanguage = (language) => {
            const snippet = codeSamples[language];
            if (!snippet) {
                console.warn(`No code sample registered for language "${language}".`);
                return;
            }
            
            languageTabButtons.forEach((button) => {
                const isActive = button.dataset.language === language;
                button.classList.toggle('active', isActive);
                button.setAttribute('aria-selected', String(isActive));
                button.setAttribute('tabindex', isActive ? '0' : '-1');
            });
            
            languageIconButtons.forEach((button) => {
                const isActive = button.dataset.language === language;
                button.classList.toggle('active', isActive);
                button.setAttribute('aria-pressed', String(isActive));
            });
            
            languageSnippetElement.innerHTML = renderSnippet(snippet);
        };
        
        const interactiveButtons = [...new Set([...languageTabButtons, ...languageIconButtons])];
        
        interactiveButtons.forEach((button) => {
            const { language } = button.dataset;
            if (!language || !codeSamples[language]) {
                return;
            }
            
            button.addEventListener('click', () => setLanguage(language));
            button.addEventListener('keydown', (event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    setLanguage(language);
                }
            });
        });
        
        const defaultLanguage =
            (languageTabButtons[0] && languageTabButtons[0].dataset.language) ||
            (languageIconButtons[0] && languageIconButtons[0].dataset.language) ||
            'python';
        
        setLanguage(defaultLanguage);

        if (languageCopyBtn) {
            languageCopyBtn.addEventListener('click', async () => {
                const codeText = languageSnippetElement ? languageSnippetElement.textContent.trim() : '';
                
                if (!codeText) {
                    return;
                }

                const showToast = () => {
                    if (!languageCopyToast) return;
                    languageCopyToast.textContent = 'Code copied to clipboard';
                    languageCopyToast.classList.add('is-visible');
                    if (copyToastTimeout) {
                        clearTimeout(copyToastTimeout);
                    }
                    copyToastTimeout = setTimeout(() => {
                        languageCopyToast.classList.remove('is-visible');
                    }, 2000);
                };

                try {
                    if (navigator.clipboard && navigator.clipboard.writeText) {
                        await navigator.clipboard.writeText(codeText);
                    } else {
                        const textArea = document.createElement('textarea');
                        textArea.value = codeText;
                        textArea.style.position = 'fixed';
                        textArea.style.top = '-1000px';
                        textArea.style.left = '-1000px';
                        document.body.appendChild(textArea);
                        textArea.focus();
                        textArea.select();
                        document.execCommand('copy');
                        document.body.removeChild(textArea);
                    }
                    showToast();
                } catch (error) {
                    console.error('Failed to copy code snippet:', error);
                }
            });
        }
    }
    
    window.addEventListener('scroll', () => {
        let current = '';
        const scrollPosition = window.pageYOffset || window.scrollY;
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;
        
        // Check if we're near the bottom of the page (within 100px)
        const isNearBottom = scrollPosition + windowHeight >= documentHeight - 100;
        
        if (isNearBottom && sections.length > 0) {
            // If near bottom, activate the last section
            const lastSection = sections[sections.length - 1];
            current = lastSection.getAttribute('id');
        } else {
            // Otherwise, find the current section based on scroll position
            sections.forEach(section => {
                const sectionTop = section.offsetTop;
                const sectionHeight = section.clientHeight;
                if (scrollPosition >= sectionTop - 300) {
                    current = section.getAttribute('id');
                }
            });
        }
        
        navDots.forEach(dot => {
            dot.classList.remove('active');
            if (dot.getAttribute('href') === `#${current}`) {
                dot.classList.add('active');
            }
        });
    });
});

// Import helper functions and ONNX Runtime at the top
// import {
//     sampleNoisyLatent,
//     loadOnnxAll,
//     loadCfgs,
//     loadProcessors,
//     loadWavRef,
//     arrayToTensor,
//     intArrayToTensor,
//     writeWavFile
// } from './helper.js';

// import * as ort from 'onnxruntime-web';

// TTS Demo functionality
(async function() {
    // Check if we're on a page with the TTS demo
    const demoTextInput = document.getElementById('demoTextInput');
    if (!demoTextInput) return;
    
    // Configure ONNX Runtime for WebGPU support
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/';
    ort.env.wasm.numThreads = 1;
    

    // Configuration
    const REF_EMBEDDING_PATHS = {
        'F': 'assets/voice_styles/F.json',
        'M': 'assets/voice_styles/M.json'
    };

    // Global state
    let models = null;
    let cfgs = null;
    let processors = null;
    let currentVoice = 'F'; // Default to Female voice
    let refEmbeddingCache = {}; // Cache for embeddings
    let currentStyleTtlTensor = null;
    let currentStyleDpTensor = null;

    // UI Elements
    const demoStatusBox = document.getElementById('demoStatusBox');
    const demoStatusText = document.getElementById('demoStatusText');
    const demoBackendBadge = document.getElementById('demoBackendBadge');
    const demoGenerateBtn = document.getElementById('demoGenerateBtn');
    const demoTotalSteps = document.getElementById('demoTotalSteps');
    const demoDurationFactor = document.getElementById('demoDurationFactor');
    const demoTotalStepsValue = document.getElementById('demoTotalStepsValue');
    const demoDurationFactorValue = document.getElementById('demoDurationFactorValue');
    const demoResults = document.getElementById('demoResults');
    const demoError = document.getElementById('demoError');
    const demoCharCount = document.getElementById('demoCharCount');
    const demoCharCounter = document.getElementById('demoCharCounter');
    const demoCharStatus = document.getElementById('demoCharStatus');
    const demoElevenLabsApiKey = document.getElementById('demoElevenLabsApiKey');
    const demoSecondaryApiKey = document.getElementById('demoSecondaryApiKey');
    const demoTertiaryApiKey = document.getElementById('demoTertiaryApiKey');
    const demoComparisonSection = document.getElementById('demoComparisonSection');
    
    // Billing Modal Elements
    const billingModal = document.getElementById('billingModal');
    const billingModalMessage = document.getElementById('billingModalMessage');
    const billingCharCount = document.getElementById('billingCharCount');
    const billingProviders = document.getElementById('billingProviders');
    const billingModalCancel = document.getElementById('billingModalCancel');
    const billingModalConfirm = document.getElementById('billingModalConfirm');

    // Text validation constants
    const MIN_CHARS = 10;
    const MAX_CHUNK_LENGTH = 300; // Maximum length for each chunk
    
    // Custom audio player state (shared across generations)
    let audioContext = null;
    let scheduledSources = [];
    let audioChunks = [];
    let totalDuration = 0;
    let startTime = 0;
    let pauseTime = 0;
    let isPaused = false;
    let isPlaying = false;
    let animationFrameId = null;
    let playPauseBtn = null;
    let progressBar = null;
    let currentTimeDisplay = null;
    let durationDisplay = null;
    let progressFill = null;
    let firstChunkGenerationTime = 0; // Processing time for first chunk
    let totalChunks = 0;
    let nextScheduledTime = 0; // Next time to schedule audio chunk
    let currentGenerationTextLength = 0;
    let supertonicPlayerRecord = null; // Supertonic player record for cross-player pause management
    let isGenerating = false; // Track if speech generation is in progress
    
    // Track all custom audio players (for ElevenLabs, etc.)
    let customAudioPlayers = [];
    const textHandlingAudioPlayers = [];
    const TEXT_HANDLING_CARD_AUDIO_MAP = [1, 2, 3, 4];
    let isComparisonMode = false;

    const isMobileViewport = () => window.matchMedia('(max-width: 768px)').matches;
    const trimDecimalsForMobile = (formatted) => {
        if (!formatted) return formatted;
        return isMobileViewport() ? formatted.replace(/\.\d{2}$/, '') : formatted;
    };

    function pauseAllPlayersExcept(currentPlayer) {
        customAudioPlayers.forEach(player => {
            if (player !== currentPlayer && player && typeof player.pausePlayback === 'function') {
                player.pausePlayback();
            }
        });
    }

    function pauseTextHandlingPlayersExcept(currentPlayer) {
        textHandlingAudioPlayers.forEach(player => {
            if (player !== currentPlayer && player && typeof player.pausePlayback === 'function') {
                player.pausePlayback();
            }
        });
    }


    /**
     * Chunk text into smaller pieces based on sentence boundaries
     * @param {string} text - The text to chunk
     * @param {number} maxLen - Maximum length for each chunk
     * @returns {Array<string>} - Array of text chunks
     */
    function chunkText(text, maxLen = MAX_CHUNK_LENGTH) {
        // Split by paragraph (two or more newlines)
        const paragraphs = text.trim().split(/\n\s*\n+/).filter(p => p.trim());
        
        const chunks = [];
        
        for (let paragraph of paragraphs) {
            paragraph = paragraph.trim();
            if (!paragraph) continue;
            
            // Split by sentence boundaries (period, question mark, exclamation mark followed by space)
            // But exclude common abbreviations like Mr., Mrs., Dr., etc. and single capital letters like F.
            const sentences = paragraph.split(/(?<!Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|Sr\.|Jr\.|Ph\.D\.|etc\.|e\.g\.|i\.e\.|vs\.|Inc\.|Ltd\.|Co\.|Corp\.|St\.|Ave\.|Blvd\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+/);
            
            let currentChunk = "";
            
            for (let sentence of sentences) {
                if (currentChunk.length + sentence.length + 1 <= maxLen) {
                    currentChunk += (currentChunk ? " " : "") + sentence;
                } else {
                    if (currentChunk) {
                        chunks.push(currentChunk.trim());
                    }
                    currentChunk = sentence;
                }
            }
            
            if (currentChunk) {
                chunks.push(currentChunk.trim());
            }
        }
        
        return chunks;
    }

    function showDemoStatus(message, type = 'info', progress = null) {
        demoStatusText.innerHTML = message;
        demoStatusBox.className = 'demo-status-box';
        demoStatusBox.style.removeProperty('--status-progress');
        demoStatusBox.style.display = ''; // Show the status box
        
        if (type === 'success') {
            demoStatusBox.classList.add('success');
        } else if (type === 'error') {
            demoStatusBox.classList.add('error');
        }
        
        // Update progress bar
        if (progress !== null && progress >= 0 && progress <= 100) {
            const clampedProgress = Math.max(0, Math.min(progress, 100));
            demoStatusBox.style.setProperty('--status-progress', `${clampedProgress}%`);
            demoStatusBox.classList.toggle('complete', clampedProgress >= 100);
        } else if (type === 'success' || type === 'error') {
            demoStatusBox.style.removeProperty('--status-progress');
            demoStatusBox.classList.remove('complete');
        } else {
            demoStatusBox.style.removeProperty('--status-progress');
            demoStatusBox.classList.remove('complete');
        }
    }

    function hideDemoStatus() {
        demoStatusBox.style.display = 'none';
    }

    function showDemoError(message) {
        demoError.textContent = message;
        demoError.classList.add('active');
    }

    function hideDemoError() {
        demoError.classList.remove('active');
    }
    
    // Custom billing confirmation modal
    function showBillingConfirmation(charCount, providers) {
        return new Promise((resolve) => {
            // Set modal content
            billingCharCount.textContent = charCount;
            billingProviders.textContent = providers.join(', ');
            billingModalMessage.textContent = 'You are about to generate speech using API services.';
            
            // Show modal
            billingModal.classList.add('show');
            
            // Handle confirm
            const handleConfirm = () => {
                cleanup();
                resolve(true);
            };
            
            // Handle cancel
            const handleCancel = () => {
                cleanup();
                resolve(false);
            };
            
            // Handle overlay click
            const handleOverlayClick = (e) => {
                if (e.target === billingModal || e.target.classList.contains('billing-modal-overlay')) {
                    cleanup();
                    resolve(false);
                }
            };
            
            // Handle escape key
            const handleEscape = (e) => {
                if (e.key === 'Escape') {
                    cleanup();
                    resolve(false);
                }
            };
            
            // Cleanup function
            const cleanup = () => {
                billingModal.classList.remove('show');
                billingModalConfirm.removeEventListener('click', handleConfirm);
                billingModalCancel.removeEventListener('click', handleCancel);
                billingModal.removeEventListener('click', handleOverlayClick);
                document.removeEventListener('keydown', handleEscape);
            };
            
            // Add event listeners
            billingModalConfirm.addEventListener('click', handleConfirm);
            billingModalCancel.addEventListener('click', handleCancel);
            billingModal.addEventListener('click', handleOverlayClick);
            document.addEventListener('keydown', handleEscape);
        });
    }

    function showBackendBadge(backend) {
        demoBackendBadge.textContent = backend;
        demoBackendBadge.classList.add('visible');
        if (backend === 'WebGPU') {
            demoBackendBadge.classList.add('webgpu');
        } else {
            demoBackendBadge.classList.add('wasm');
        }
    }

    // Validate characters in text
    function validateCharacters(text) {
        if (!processors || !processors.textProcessor) {
            return { valid: true, unsupportedChars: [] };
        }
        
        try {
            // Extract unique characters to minimize preprocessText calls
            const uniqueChars = [...new Set(text)];
            
            // Build mapping for unique chars only (much faster for long texts)
            // For example, Korean '간' -> 'ㄱㅏㄴ', so we map 'ㄱ','ㅏ','ㄴ' -> '간'
            const processedToOriginal = new Map();
            const charToProcessed = new Map();
            
            for (const char of uniqueChars) {
                const processedChar = preprocessText(char);
                charToProcessed.set(char, processedChar);
                
                // Map each processed character back to its original
                for (const pc of processedChar) {
                    if (!processedToOriginal.has(pc)) {
                        processedToOriginal.set(pc, new Set());
                    }
                    processedToOriginal.get(pc).add(char);
                }
            }
            
            // Build full processed text using cached mappings
            const fullProcessedText = Array.from(text).map(c => charToProcessed.get(c)).join('');
            
            // Check the entire processed text once (efficient)
            const { unsupportedChars } = processors.textProcessor.call([fullProcessedText]);
            
            // Map unsupported processed chars back to original chars
            const unsupportedOriginalChars = new Set();
            if (unsupportedChars && unsupportedChars.length > 0) {
                for (const unsupportedChar of unsupportedChars) {
                    const originalChars = processedToOriginal.get(unsupportedChar);
                    if (originalChars) {
                        originalChars.forEach(c => unsupportedOriginalChars.add(c));
                    }
                }
            }
            
            const unsupportedCharsArray = Array.from(unsupportedOriginalChars);
            return { 
                valid: unsupportedCharsArray.length === 0, 
                unsupportedChars: unsupportedCharsArray
            };
        } catch (error) {
            return { valid: true, unsupportedChars: [] };
        }
    }

    // Update character counter and validate text length
    function updateCharCounter() {
        const text = demoTextInput.value;
        const length = text.length;
        
        demoCharCount.textContent = length;
        
        // Get the actual width of the textarea
        const textareaWidth = demoTextInput.offsetWidth;
        // Max width reference: 1280px (container max-width) / 2 (grid column) - padding/gap ≈ 638px
        // Using 640px as reference for easier calculation
        const maxWidthRef = 640;
        
        // Calculate font size based on width ratio
        // Original rem values at max-width (640px):
        // 5rem = 80px @ 16px base → 80/640 = 12.5%
        // 4rem = 64px → 64/640 = 10%
        // 3rem = 48px → 48/640 = 7.5%
        // 2.5rem = 40px → 40/640 = 6.25%
        // 2rem = 32px → 32/640 = 5%
        // 1.5rem = 24px → 24/640 = 3.75%
        // 1rem = 16px → 16/640 = 2.5%
        
        let fontSizeRatio;
        if (length < 160) {
            fontSizeRatio = 0.06375; // ~6.375% of width (scaled from 3rem)
        } else if (length < 240) {
            fontSizeRatio = 0.053125; // ~5.3125% of width (scaled from 2.5rem)
        } else if (length < 400) {
            fontSizeRatio = 0.0425; // ~4.25% of width (scaled from 2rem)
        } else if (length < 700) {
            fontSizeRatio = 0.031875; // ~3.1875% of width (scaled from 1.5rem)
        } else {
            fontSizeRatio = 0.025; // 2.5% of width (minimum stays the same)
        }
        
        // Calculate font size based on actual width
        const fontSize = textareaWidth * fontSizeRatio;
        demoTextInput.style.fontSize = `${fontSize}px`;
        
        // Remove all status classes
        demoCharCounter.classList.remove('error', 'warning', 'valid');
        
        // Check for unsupported characters first (only if models are loaded)
        let hasUnsupportedChars = false;
        if (models && processors && length > 0) {
            const validation = validateCharacters(text);
            if (!validation.valid && validation.unsupportedChars.length > 0) {
                hasUnsupportedChars = true;
                const charList = validation.unsupportedChars.slice(0, 5).map(c => `"${c}"`).join(', ');
                const moreChars = validation.unsupportedChars.length > 5 ? ` and ${validation.unsupportedChars.length - 5} more` : '';
                showDemoError(`Unsupported characters detected: ${charList}${moreChars}. Please remove them before generating speech.`);
            } else {
                hideDemoError();
            }
        }
        
        // Update status based on length and character validation
        if (length < MIN_CHARS) {
            demoCharCounter.classList.add('error');
            demoCharStatus.textContent = '✗';
            demoGenerateBtn.disabled = true;
        } else if (hasUnsupportedChars) {
            demoCharCounter.classList.add('error');
            demoCharStatus.textContent = '✗';
            demoGenerateBtn.disabled = true;
        } else {
            demoCharCounter.classList.add('valid');
            demoCharStatus.textContent = '✓';
            // Enable only if models are loaded AND not currently generating
            demoGenerateBtn.disabled = !models || isGenerating;
        }
    }

    // Validate text input
    function validateTextInput(text) {
        if (!text || text.trim().length === 0) {
            return { valid: false, message: 'Please enter some text.' };
        }
        if (text.length < MIN_CHARS) {
            return { valid: false, message: `Text must be at least ${MIN_CHARS} characters long. (Currently ${text.length})` };
        }
        return { valid: true };
    }

    // Load pre-extracted style embeddings from JSON
    async function loadStyleEmbeddings(voice) {
        try {
            // Check if already cached
            if (refEmbeddingCache[voice]) {
                return refEmbeddingCache[voice];
            }
            
            const embeddingPath = REF_EMBEDDING_PATHS[voice];
            if (!embeddingPath) {
                throw new Error(`No embedding path configured for voice: ${voice}`);
            }
            
            const response = await fetch(embeddingPath);
            if (!response.ok) {
                throw new Error(`Failed to fetch embedding: ${response.statusText}`);
            }
            
            const embeddingData = await response.json();
            
            // Convert JSON data to ONNX tensors
            // Flatten nested arrays before creating Float32Array
            const styleTtlData = embeddingData.style_ttl.data.flat(Infinity);
            const styleTtlTensor = new ort.Tensor(
                embeddingData.style_ttl.type || 'float32',
                Float32Array.from(styleTtlData),
                embeddingData.style_ttl.dims
            );
            
            const styleDpData = embeddingData.style_dp.data.flat(Infinity);
            const styleDpTensor = new ort.Tensor(
                embeddingData.style_dp.type || 'float32',
                Float32Array.from(styleDpData),
                embeddingData.style_dp.dims
            );
            
            const embeddings = {
                styleTtl: styleTtlTensor,
                styleDp: styleDpTensor
            };
            
            // Cache the embeddings
            refEmbeddingCache[voice] = embeddings;
            
            return embeddings;
        } catch (error) {
            throw error;
        }
    }
    
    // Switch to a different voice
    async function switchVoice(voice) {
        try {
            const embeddings = await loadStyleEmbeddings(voice);
            
            currentStyleTtlTensor = embeddings.styleTtl;
            currentStyleDpTensor = embeddings.styleDp;
            currentVoice = voice;
            
            // Re-validate text after switching voice
            updateCharCounter();
        } catch (error) {
            showDemoError(`Failed to load ${voice === 'F' ? 'Female' : 'Male'} voice: ${error.message}`);
            throw error;
        }
    }

    // Check WebGPU support more thoroughly
    async function checkWebGPUSupport() {
        try {
            // Detect iOS/Safari
            const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) || 
                          (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
            
            // iOS and Safari have incomplete WebGPU support
            if (isIOS) {
                return { supported: false, reason: 'iOS does not support the required WebGPU features' };
            }
            
            // Check if WebGPU is available in the browser
            if (!navigator.gpu) {
                return { supported: false, reason: 'WebGPU not available in this browser' };
            }
            
            // Request adapter
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                return { supported: false, reason: 'No WebGPU adapter found' };
            }
            
            // Check adapter info
            try {
                const adapterInfo = await adapter.requestAdapterInfo();
            } catch (infoError) {
                // Ignore adapter info errors
            }
            
            // Request device to test if it actually works
            const device = await adapter.requestDevice();
            if (!device) {
                return { supported: false, reason: 'Failed to create WebGPU device' };
            }
            
            return { supported: true, adapter, device };
        } catch (error) {
            // Handle specific iOS/Safari errors
            const errorMsg = error.message || '';
            if (errorMsg.includes('subgroupMinSize') || errorMsg.includes('subgroup')) {
                return { supported: false, reason: 'iOS/Safari does not support required WebGPU features (subgroup operations)' };
            }
            return { supported: false, reason: error.message };
        }
    }

    // Warmup models with dummy inference (no audio playback, no UI updates)
    async function warmupModels() {
        try {
            const dummyText = 'Looking to integrate Supertonic into your product? We offer customized on-device SDK solutions tailored to your business needs. Our lightweight, high-performance TTS technology can be seamlessly integrated into mobile apps, IoT devices, automotive systems, and more. Try it now, and enjoy its speed.';
            const totalStep = 5; // Use minimal steps for faster warmup
            const durationFactor = 1.0;
            
            const textList = [dummyText];
            const bsz = 1;
            
            // Use pre-computed style embeddings
            const styleTtlTensor = currentStyleTtlTensor;
            const styleDpTensor = currentStyleDpTensor;
            
            // Step 1: Estimate duration
            const { textIds, textMask } = processors.textProcessor.call(textList);
            
            const textIdsShape = [bsz, textIds[0].length];
            const textMaskShape = [bsz, 1, textMask[0][0].length];
            const textMaskTensor = arrayToTensor(textMask, textMaskShape);
            
            const dpResult = await models.dpOrt.run({
                text_ids: intArrayToTensor(textIds, textIdsShape),
                style_dp: styleDpTensor,
                text_mask: textMaskTensor
            });
            
            const durOnnx = Array.from(dpResult.duration.data);
            for (let i = 0; i < durOnnx.length; i++) {
                durOnnx[i] *= durationFactor;
            }
            const durReshaped = [];
            for (let b = 0; b < bsz; b++) {
                durReshaped.push([[durOnnx[b]]]);
            }
            
            // Step 2: Encode text
            const textEncResult = await models.textEncOrt.run({
                text_ids: intArrayToTensor(textIds, textIdsShape),
                style_ttl: styleTtlTensor,
                text_mask: textMaskTensor
            });
            
            const textEmbTensor = textEncResult.text_emb;
            
            // Step 3: Denoising
            let { noisyLatent, latentMask } = sampleNoisyLatent(durReshaped, cfgs);
            const latentShape = [bsz, noisyLatent[0].length, noisyLatent[0][0].length];
            const latentMaskShape = [bsz, 1, latentMask[0][0].length];
            const latentMaskTensor = arrayToTensor(latentMask, latentMaskShape);
                    
            const totalStepArray = new Array(bsz).fill(totalStep);
            const scalarShape = [bsz];
            const totalStepTensor = arrayToTensor(totalStepArray, scalarShape);
            
            for (let step = 0; step < totalStep; step++) {
                const currentStepArray = new Array(bsz).fill(step);
            
                const vectorEstResult = await models.vectorEstOrt.run({
                    noisy_latent: arrayToTensor(noisyLatent, latentShape),
                    text_emb: textEmbTensor,
                    style_ttl: styleTtlTensor,
                    text_mask: textMaskTensor,
                    latent_mask: latentMaskTensor,
                    total_step: totalStepTensor,
                    current_step: arrayToTensor(currentStepArray, scalarShape)
                });
                
                const denoisedLatent = Array.from(vectorEstResult.denoised_latent.data);
                
                // Update latent
                let idx = 0;
                for (let b = 0; b < noisyLatent.length; b++) {
                    for (let d = 0; d < noisyLatent[b].length; d++) {
                        for (let t = 0; t < noisyLatent[b][d].length; t++) {
                            noisyLatent[b][d][t] = denoisedLatent[idx++];
                        }
                    }
                }
            }
            
            // Step 4: Generate waveform
            const vocoderResult = await models.vocoderOrt.run({
                latent: arrayToTensor(noisyLatent, latentShape)
            });
            
            // Warmup complete - no need to process the audio further
        } catch (error) {
            console.warn('Warmup failed (non-critical):', error.message);
            // Don't throw - warmup failure shouldn't prevent normal usage
        }
    }

    // Load models on page load
    async function initializeModels() {
        try {
            showDemoStatus('<strong>Loading configuration...</strong>', 'info', 5);
            
            const basePath = 'assets/onnx';
            
            // Load config
            cfgs = await loadCfgs(basePath);
            
            // Check WebGPU support first
            showDemoStatus('<strong>Checking WebGPU support...</strong>', 'info', 8);
            const webgpuCheck = await checkWebGPUSupport();
            
            // If WebGPU is not supported, show message and disable demo
            if (!webgpuCheck.supported) {
                // Show specific message for iOS users
                const errorMessage = webgpuCheck.reason.includes('iOS')
                    ? `<strong>iOS is not currently supported.</strong><br>Please use a desktop browser that supports WebGPU (Chrome 113+, Edge 113+).`
                    : `Please use a browser that supports WebGPU (Chrome 113+, Edge 113+, or other WebGPU-enabled browsers).`;
                
                showDemoStatus(errorMessage, 'error', 100);
                showBackendBadge('Not Supported');
                
                // Disable all input elements
                demoTextInput.disabled = true;
                demoGenerateBtn.disabled = true;
                demoTotalSteps.disabled = true;
                demoDurationFactor.disabled = true;
                demoElevenLabsApiKey.disabled = true;
                if (demoSecondaryApiKey) demoSecondaryApiKey.disabled = true;
                if (demoTertiaryApiKey) demoTertiaryApiKey.disabled = true;
                
                // Disable voice toggle
                const voiceToggleTexts = document.querySelectorAll('.voice-toggle-text');
                voiceToggleTexts.forEach(text => {
                    text.classList.add('disabled');
                    text.style.pointerEvents = 'none';
                    text.style.opacity = '0.5';
                });
                
                return; // Stop initialization
            }
            
            // Load models with WebGPU
            showDemoStatus('<strong>WebGPU detected! Loading models...</strong>', 'info', 10);
            
            const modelsLoadPromise = loadOnnxAll(basePath, {
                executionProviders: ['webgpu'],
                graphOptimizationLevel: 'all'
            }, (modelName, current, total) => {
                const progress = 10 + (current / total) * 70; // 10-80% for model loading
                showDemoStatus(`<strong>Loading models with WebGPU (${current}/${total}):</strong> ${modelName}...`, 'info', progress);
            });
            
            // Load processors in parallel with models
            const [loadedModels, loadedProcessors] = await Promise.all([
                modelsLoadPromise,
                loadProcessors(basePath)
            ]);
            
            models = loadedModels;
            processors = loadedProcessors;            
            showDemoStatus('<strong>Loading reference embeddings...</strong>', 'info', 85);
            
            // Load pre-extracted embeddings for default voice
            const embeddings = await loadStyleEmbeddings(currentVoice);
            currentStyleTtlTensor = embeddings.styleTtl;
            currentStyleDpTensor = embeddings.styleDp;
            
            showDemoStatus('<strong>Warming up models...</strong>', 'info', 90);
            
            // Warmup step: run inference once in background with dummy text
            await warmupModels();
            
            hideDemoStatus();
            demoGenerateBtn.disabled = false;
            
            // Enable voice toggle buttons after models are loaded
            const voiceToggleTexts = document.querySelectorAll('.voice-toggle-text');
            voiceToggleTexts.forEach(text => text.classList.remove('disabled'));
            
            // Validate initial text now that models are loaded
            updateCharCounter();
            
        } catch (error) {
            showDemoStatus(`<strong>Error:</strong> ${error.message}`, 'error');
            showDemoError(`Failed to initialize: ${error.message}. Check console for details.`);
        }
    }

    // ElevenLabs API synthesis function
    async function generateSpeechElevenLabs(text, apiKey) {
        const startTime = Date.now();
        
        try {
            const response = await fetch('https://api.elevenlabs.io/v1/text-to-speech/JBFqnCBsd6RMkjVDRZzb', {
                method: 'POST',
                headers: {
                    'Accept': 'audio/mpeg',
                    'Content-Type': 'application/json',
                    'xi-api-key': apiKey
                },
                body: JSON.stringify({
                    text: text,
                    model_id: 'eleven_flash_v2_5',
                    voice_settings: {
                        stability: 0.5,
                        similarity_boost: 0.5
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`ElevenLabs API error: ${response.status} ${response.statusText}`);
            }
            
            const audioBlob = await response.blob();
            const audioBuffer = await audioBlob.arrayBuffer();
            
            // Get audio duration
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const decodedAudio = await audioContext.decodeAudioData(audioBuffer);
            const audioDuration = decodedAudio.duration;
            
            const endTime = Date.now();
            const processingTime = (endTime - startTime) / 1000;
            
            return {
                success: true,
                audioBlob,
                audioDuration,
                processingTime,
                url: URL.createObjectURL(audioBlob),
                text: text  // 추가: text를 반환에 포함
            };
        } catch (error) {
            const endTime = Date.now();
            const processingTime = (endTime - startTime) / 1000;
            
            return {
                success: false,
                error: error.message,
                processingTime,
                text: text  // 추가: 에러 시에도 text 포함
            };
        }
    }

    // OpenAI TTS-1 API synthesis function
    async function generateSpeechOpenAI(text, apiKey) {
        const startTime = Date.now();
        
        try {
            const response = await fetch('https://api.openai.com/v1/audio/speech', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: 'tts-1',
                    input: text,
                    voice: 'alloy',
                    response_format: 'mp3'
                })
            });
            
            if (!response.ok) {
                throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
            }
            
            const audioBlob = await response.blob();
            const audioBuffer = await audioBlob.arrayBuffer();
            
            // Get audio duration
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const decodedAudio = await audioContext.decodeAudioData(audioBuffer);
            const audioDuration = decodedAudio.duration;
            
            const endTime = Date.now();
            const processingTime = (endTime - startTime) / 1000;
            
            return {
                success: true,
                audioBlob,
                audioDuration,
                processingTime,
                url: URL.createObjectURL(audioBlob),
                text: text
            };
        } catch (error) {
            const endTime = Date.now();
            const processingTime = (endTime - startTime) / 1000;
            
            return {
                success: false,
                error: error.message,
                processingTime,
                text: text
            };
        }
    }

    // Gemini 2.5 Flash TTS API synthesis function
    async function generateSpeechGemini(text, apiKey) {
        const startTime = Date.now();
        
        try {
            const response = await fetch('https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent', {
                method: 'POST',
                headers: {
                    'x-goog-api-key': apiKey,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    contents: [{
                        parts: [{
                            text: text
                        }]
                    }],
                    generationConfig: {
                        responseModalities: ["AUDIO"],
                        speechConfig: {
                            voiceConfig: {
                                prebuiltVoiceConfig: {
                                    voiceName: "Kore"
                                }
                            }
                        }
                    }
                })
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Gemini API error: ${response.status} ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Extract audio data from Gemini response
            let audioContent = null;
            let mimeType = null;
            
            if (data.candidates && data.candidates[0]?.content?.parts) {
                for (const part of data.candidates[0].content.parts) {
                    if (part.inlineData && part.inlineData.data) {
                        audioContent = part.inlineData.data;
                        mimeType = part.inlineData.mimeType;
                        break;
                    }
                }
            }
            
            if (!audioContent) {
                throw new Error('No audio content found in Gemini response');
            }
            
            // Decode base64 audio content
            const binaryString = atob(audioContent);
            const pcmData = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                pcmData[i] = binaryString.charCodeAt(i);
            }
            
            // Parse sample rate from mimeType if available (e.g., "audio/pcm;rate=24000")
            let sampleRate = 24000; // default
            if (mimeType && mimeType.includes('rate=')) {
                const match = mimeType.match(/rate=(\d+)/);
                if (match) {
                    sampleRate = parseInt(match[1]);
                }
            }
            
            // Gemini returns s16le (signed 16-bit little-endian PCM)
            const numChannels = 1; // mono
            const bitsPerSample = 16;
            const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
            const blockAlign = numChannels * (bitsPerSample / 8);
            const dataSize = pcmData.length;
            
            // Create WAV header (44 bytes)
            const wavHeader = new ArrayBuffer(44);
            const view = new DataView(wavHeader);
            
            // RIFF chunk descriptor
            view.setUint32(0, 0x52494646, false); // "RIFF"
            view.setUint32(4, 36 + dataSize, true); // File size - 8
            view.setUint32(8, 0x57415645, false); // "WAVE"
            
            // fmt sub-chunk
            view.setUint32(12, 0x666d7420, false); // "fmt "
            view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
            view.setUint16(20, 1, true); // AudioFormat (1 for PCM)
            view.setUint16(22, numChannels, true); // NumChannels
            view.setUint32(24, sampleRate, true); // SampleRate
            view.setUint32(28, byteRate, true); // ByteRate
            view.setUint16(32, blockAlign, true); // BlockAlign
            view.setUint16(34, bitsPerSample, true); // BitsPerSample
            
            // data sub-chunk
            view.setUint32(36, 0x64617461, false); // "data"
            view.setUint32(40, dataSize, true); // Subchunk2Size
            
            // Combine header and PCM data
            const wavData = new Uint8Array(44 + dataSize);
            wavData.set(new Uint8Array(wavHeader), 0);
            wavData.set(pcmData, 44);
            
            const finalAudioBuffer = wavData.buffer;
            
            // Get audio duration
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            let decodedAudio;
            try {
                decodedAudio = await audioContext.decodeAudioData(finalAudioBuffer.slice(0)); // Use slice to create a copy
            } catch (decodeError) {
                throw new Error(`Unable to decode Gemini audio: ${decodeError.message}`);
            }
            const audioDuration = decodedAudio.duration;
            await audioContext.close();
            
            // Create blob with WAV format
            const audioBlob = new Blob([finalAudioBuffer], { type: 'audio/wav' });
            
            const endTime = Date.now();
            const processingTime = (endTime - startTime) / 1000;
            
            return {
                success: true,
                audioBlob,
                audioDuration,
                processingTime,
                url: URL.createObjectURL(audioBlob),
                text: text
            };
        } catch (error) {
            const endTime = Date.now();
            const processingTime = (endTime - startTime) / 1000;
            
            return {
                success: false,
                error: error.message,
                processingTime,
                text: text
            };
        }
    }

    // Update individual system result in comparison table
    function updateComparisonRow(system, result) {
        if (!isComparisonMode) return;
        const statusEl = document.getElementById(`${system}Status`);
        const titleStatusEl = document.getElementById(`${system}-status`);
        const timeEl = document.getElementById(`${system}Time`);
        const durationEl = document.getElementById(`${system}Duration`);
        const rtfEl = document.getElementById(`${system}RTF`);
        
        if (result.success) {
            if (statusEl) {
                statusEl.textContent = '';
                statusEl.className = 'demo-comparison-cell';
            }
            if (titleStatusEl) {
                titleStatusEl.textContent = '✅ Completed';
                titleStatusEl.classList.remove('status-error', 'status-running');
                titleStatusEl.classList.add('status-success');
            }
            timeEl.textContent = `${result.processingTime.toFixed(2)}s`;
            durationEl.textContent = `${result.audioDuration.toFixed(2)}s`;
            
            const rtfValue = result.processingTime / result.audioDuration;
            rtfEl.innerHTML = `<strong>${rtfValue.toFixed(3)}x</strong>`;
        } else {
            if (statusEl) {
                statusEl.textContent = '';
                statusEl.className = 'demo-comparison-cell';
            }
            if (titleStatusEl) {
                titleStatusEl.textContent = '❌ Failed';
                titleStatusEl.classList.remove('status-success', 'status-running');
                titleStatusEl.classList.add('status-error');
            }
            timeEl.textContent = result.error || 'Error';
            durationEl.textContent = '-';
            rtfEl.textContent = '-';
        }
    }
    
    // Highlight winner after all complete (based on RTF)
    function highlightWinner(results) {
        if (!isComparisonMode) return;
        if (!Array.isArray(results) || results.length < 2) return;
        
        // Remove all winner classes first
        const systems = ['supertonic', 'elevenlabs', 'openai', 'gemini'];
        systems.forEach(system => {
            const row = document.querySelector(`.${system}-row`);
            const rtfEl = document.getElementById(`${system}RTF`);
            if (row) row.classList.remove('winner');
            if (rtfEl) rtfEl.classList.remove('fastest');
        });
        
        // Calculate RTF for each result and find the best one
        const systemResults = [];
        
        results.forEach((result, index) => {
            if (result && result.success && result.audioDuration > 0) {
                const rtfValue = result.processingTime / result.audioDuration;
                // Determine system by result order in the results array
                // Results are typically passed in order: [supertonicResult, elevenlabsResult, openaiResult, geminiResult]
                let system = null;
                if (index === 0 || result.text === results[0]?.text) {
                    system = 'supertonic';
                } else {
                    // Check which system by looking at existing elements
                    const hasElevenlabs = document.querySelector('.elevenlabs-row');
                    const hasOpenai = document.querySelector('.openai-row');
                    const hasGemini = document.querySelector('.gemini-row');
                    
                    if (hasElevenlabs && !systemResults.find(s => s.system === 'elevenlabs')) {
                        system = 'elevenlabs';
                    } else if (hasOpenai && !systemResults.find(s => s.system === 'openai')) {
                        system = 'openai';
                    } else if (hasGemini && !systemResults.find(s => s.system === 'gemini')) {
                        system = 'gemini';
                    }
                }
                
                if (system) {
                    systemResults.push({ system, rtfValue });
                }
            }
        });
        
        // Find the best (lowest RTF)
        if (systemResults.length > 0) {
            const best = systemResults.reduce((prev, curr) => 
                curr.rtfValue < prev.rtfValue ? curr : prev
            );
            
            const row = document.querySelector(`.${best.system}-row`);
            const rtfEl = document.getElementById(`${best.system}RTF`);
            if (row) row.classList.add('winner');
            if (rtfEl) rtfEl.classList.add('fastest');
        }
    }

    // Supertonic synthesis function (extracted for parallel execution)
    async function generateSupertonicSpeech(text, totalStep, durationFactor) {
        const supertonicStartTime = Date.now();
        
        try {
            const textList = [text];
            const bsz = 1;
            const sampleRate = cfgs.ae.sample_rate;
            
            // Use pre-computed style embeddings
            const styleTtlTensor = currentStyleTtlTensor;
            const styleDpTensor = currentStyleDpTensor;
            
            // Step 1: Estimate duration
            const { textIds, textMask, unsupportedChars } = processors.textProcessor.call(textList);
            
            // Check for unsupported characters
            if (unsupportedChars && unsupportedChars.length > 0) {
                const charList = unsupportedChars.map(c => `"${c}"`).join(', ');
                throw new Error(`Unsupported characters: ${charList}`);
            }
            
            const textIdsShape = [bsz, textIds[0].length];
            const textMaskShape = [bsz, 1, textMask[0][0].length];
            const textMaskTensor = arrayToTensor(textMask, textMaskShape);
            
            const dpResult = await models.dpOrt.run({
                text_ids: intArrayToTensor(textIds, textIdsShape),
                style_dp: styleDpTensor,
                text_mask: textMaskTensor
            });
            
            const durOnnx = Array.from(dpResult.duration.data);
            // Apply duration factor to adjust speech length (once)
            const durationAdjustment = currentVoice === 'F' ? 0.1 : 0.08;
            for (let i = 0; i < durOnnx.length; i++) {
                durOnnx[i] *= (durationFactor - durationAdjustment);
            }
            const durReshaped = [];
            for (let b = 0; b < bsz; b++) {
                durReshaped.push([[durOnnx[b]]]);
            }
            
            // Step 2: Encode text
            const textEncResult = await models.textEncOrt.run({
                text_ids: intArrayToTensor(textIds, textIdsShape),
                style_ttl: styleTtlTensor,
                text_mask: textMaskTensor
            });
            
            const textEmbTensor = textEncResult.text_emb;
            
            // Step 3: Denoising
            let { noisyLatent, latentMask } = sampleNoisyLatent(durReshaped, cfgs);
            const latentShape = [bsz, noisyLatent[0].length, noisyLatent[0][0].length];
            const latentMaskShape = [bsz, 1, latentMask[0][0].length];
            const latentMaskTensor = arrayToTensor(latentMask, latentMaskShape);
                    
            // Prepare constant tensors
            const totalStepArray = new Array(bsz).fill(totalStep);
            const scalarShape = [bsz];
            const totalStepTensor = arrayToTensor(totalStepArray, scalarShape);
            
            for (let step = 0; step < totalStep; step++) {
                const currentStepArray = new Array(bsz).fill(step);
            
                const vectorEstResult = await models.vectorEstOrt.run({
                    noisy_latent: arrayToTensor(noisyLatent, latentShape),
                    text_emb: textEmbTensor,
                    style_ttl: styleTtlTensor,
                    text_mask: textMaskTensor,
                    latent_mask: latentMaskTensor,
                    total_step: totalStepTensor,
                    current_step: arrayToTensor(currentStepArray, scalarShape)
                });
                
                const denoisedLatent = Array.from(vectorEstResult.denoised_latent.data);
                
                // Update latent
                let idx = 0;
                for (let b = 0; b < noisyLatent.length; b++) {
                    for (let d = 0; d < noisyLatent[b].length; d++) {
                        for (let t = 0; t < noisyLatent[b][d].length; t++) {
                            noisyLatent[b][d][t] = denoisedLatent[idx++];
                        }
                    }
                }
            }
            
            // Step 4: Generate waveform
            const vocoderResult = await models.vocoderOrt.run({
                latent: arrayToTensor(noisyLatent, latentShape)
            });
            
            const wavBatch = Array.from(vocoderResult.wav_tts.data);
            const wavLen = Math.floor(sampleRate * durOnnx[0]);
            const wavOut = wavBatch.slice(0, wavLen);
            
            // Create WAV file
            const wavBuffer = writeWavFile(wavOut, sampleRate);
            const blob = new Blob([wavBuffer], { type: 'audio/wav' });
            const url = URL.createObjectURL(blob);
            
            // Calculate times for Supertonic
            const supertonicEndTime = Date.now();
            const supertonicProcessingTime = (supertonicEndTime - supertonicStartTime) / 1000;
            const audioDurationSec = durOnnx[0];
            
            return {
                success: true,
                processingTime: supertonicProcessingTime,
                audioDuration: audioDurationSec,
                url: url,
                text: text
            };
        } catch (error) {
            return {
                success: false,
                error: error.message,
                text: text
            };
        }
    }

    // Format time: 60초 미만 -> 00.00, 60분 미만 -> 00:00.00, 60분 이상 -> 00:00:00.00
    function formatTimeDetailed(seconds) {
        const hours = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        const ms = Math.floor((secs % 1) * 100);
        const wholeSecs = Math.floor(secs);
        
        if (seconds < 60) {
            return `${wholeSecs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
        } else if (seconds < 3600) {
            return `${mins.toString().padStart(2, '0')}:${wholeSecs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
        } else {
            return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${wholeSecs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
        }
    }

    // Render result to UI with custom audio player
    async function renderResult(system, result, isFirst = false) {
        const container = document.getElementById('demoResults');
        
        const formatTime = (seconds, { trimMobile = false } = {}) => {
            const mins = Math.floor(seconds / 60);
            const secs = seconds % 60;
            const secString = secs.toFixed(2).padStart(5, '0');
            let formatted = `${mins}:${secString}`;
            if (trimMobile) {
                formatted = trimDecimalsForMobile(formatted);
            }
            return formatted;
        };
        
        const textLength = result.text ? result.text.length : 0;
        const isBatch = textLength >= MAX_CHUNK_LENGTH;
        const successfulResult = result && result.success;
        const firstChunkTimeValue = result.firstChunkTime;
        const processingTimeStr = successfulResult
            ? (isBatch && firstChunkTimeValue
                ? `${formatTimeDetailed(firstChunkTimeValue)} / ${formatTimeDetailed(result.processingTime)}`
                : formatTimeDetailed(result.processingTime))
            : (result.error || 'Error');
        const charsPerSec = successfulResult && result.processingTime > 0 ? (textLength / result.processingTime).toFixed(1) : '-';
        const rtf = successfulResult && result.audioDuration > 0 ? (result.processingTime / result.audioDuration).toFixed(3) : '-';
        const progressValue = successfulResult && textLength > 0 ? 100 : 0;
        const titleMain = system === 'supertonic' ? 'Supertonic' : 
                         (system === 'openai' ? 'OpenAI TTS-1' : 
                         (system === 'gemini' ? 'Gemini 2.5 Flash TTS' : 'ElevenLabs Flash v2.5'));
        const titleSub = system === 'supertonic' ? 'On-Device' : 'Cloud API';
        const titleColor =
            system === 'supertonic'
                ? 'var(--supertone_blue)'
                : system === 'elevenlabs'
                    ? 'var(--brand-elevenlabs)'
                    : system === 'openai'
                        ? 'var(--brand-openai)'
                        : system === 'gemini'
                            ? 'var(--brand-gemini)'
                            : '#999';
        const titleStatus = isComparisonMode
            ? `<span class="title-status status-running" id="${system}-status">⏳ Running...</span>`
            : '';
        const hasAudio = successfulResult && result.url;
        const totalDurationDisplay = successfulResult && typeof result.audioDuration === 'number'
            ? formatTime(result.audioDuration, { trimMobile: true })
            : '--';
        const downloadActionsHTML = hasAudio ? `
                    <div class="demo-result-actions">
                        <button class="demo-download-btn" onclick="downloadDemoAudio('${result.url}', '${system}_speech.${system === 'supertonic' ? 'wav' : 'mp3'}')" aria-label="Download ${system === 'supertonic' ? 'WAV' : 'MP3'}" title="Download ${system === 'supertonic' ? 'WAV' : 'MP3'}">
                            <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                                <polyline points="7 10 12 15 17 10"/>
                                <line x1="12" y1="15" x2="12" y2="3"/>
                            </svg>
                        </button>
                    </div>` : '';

        const infoMarkupSuccess = `
                    <!--
                    <div class="stat">
                        <div class="stat-value" id="${system}-chars">${textLength}</div>
                        <div class="stat-label">Processed Chars</div>
                    </div>
                    -->
                    <div class="stat">
                        <div class="stat-value" id="${system}-time">${formatStatValueWithSuffix(processingTimeStr, 's', { firstLabel: true })}</div>
                        <div class="stat-label">Processing Time<span class="stat-arrow stat-arrow--down">↓</span></div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="${system}-cps">${charsPerSec}</div>
                        <div class="stat-label">Chars/sec<span class="stat-arrow stat-arrow--up">↑</span></div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="${system}-rtf">${formatStatValueWithSuffix(rtf, 'x')}</div>
                        <div class="stat-label">RTF<span class="stat-arrow stat-arrow--down">↓</span></div>
                    </div>
                    <!--
                    <div class="stat">
                        <div class="stat-value">${progressValue.toFixed ? progressValue.toFixed(1) : progressValue}%</div>
                        <div class="stat-label">Progress</div>
                    </div>
                    -->
        `;
        const infoMarkupError = `
                    <div class="stat" style="width: 100%;">
                        <div class="stat-value">${result.error || 'Failed'}</div>
                    </div>
        `;
        const resultItemEl = document.getElementById(`${system}-result`);
        const infoContainer = resultItemEl ? resultItemEl.querySelector('.demo-result-info') : null;
        const playerContainer = resultItemEl ? resultItemEl.querySelector('.custom-audio-player') : null;

        if (resultItemEl && infoContainer && playerContainer) {
            resultItemEl.classList.add(`${system}-result-item`);
            resultItemEl.classList.remove('generating');
            resultItemEl.style.setProperty('--result-progress', `${progressValue}%`);
            resultItemEl.style.setProperty('--provider-color', titleColor);

            const titleMainEl = resultItemEl.querySelector('.title-main');
            if (titleMainEl) {
                titleMainEl.textContent = titleMain;
                titleMainEl.style.color = titleColor;
            }
            const titleSubEl = resultItemEl.querySelector('.title-sub');
            if (titleSubEl) {
                titleSubEl.textContent = titleSub;
            }
            if (!resultItemEl.querySelector('.title-status') && titleStatus) {
                const titleEl = resultItemEl.querySelector('.demo-result-title');
                if (titleEl) {
                    titleEl.insertAdjacentHTML('beforeend', titleStatus);
                }
            }

            infoContainer.classList.toggle('error', !successfulResult);
            infoContainer.innerHTML = successfulResult ? infoMarkupSuccess : infoMarkupError;

            if (successfulResult) {
                playerContainer.style.display = '';
                playerContainer.innerHTML = `
                    <button id="${system}-play-pause-btn" class="player-btn"${hasAudio ? '' : ' disabled'}>${hasAudio ? PLAY_ICON_SVG : STOP_ICON_SVG}</button>
                    <div class="time-display" id="${system}-current-time">0:00.00</div>
                    <div class="progress-container" id="${system}-progress-container">
                        <div class="progress-bar">
                            <div class="progress-fill" id="${system}-progress-fill"></div>
                        </div>
                    </div>
                    <div class="time-display" id="${system}-total-duration">${totalDurationDisplay}</div>
                    ${downloadActionsHTML}
                `;
            } else {
                playerContainer.style.display = 'none';
                playerContainer.innerHTML = '';
            }

            container.style.display = 'flex';

            if (successfulResult && hasAudio) {
                await setupCustomPlayer(system, result);
            } else if (successfulResult && !hasAudio) {
                const playBtnEl = document.getElementById(`${system}-play-pause-btn`);
                if (playBtnEl) playBtnEl.disabled = true;
            }
            return;
        }

        const infoSection = successfulResult
            ? `<div class="demo-result-info">${infoMarkupSuccess}</div>`
            : `<div class="demo-result-info error">${infoMarkupError}</div>`;
        
        const resultHTML = `
            <div class="demo-result-item ${system}-result-item generating" id="${system}-result" style="--result-progress: ${progressValue}%">
                <div class="demo-result-title">
                    <span class="title-main" style="color: ${titleColor};">${titleMain}</span>
                    <span class="title-sub">${titleSub}</span>
                    ${titleStatus}
                </div>
                ${infoSection}
                <div class="custom-audio-player">
                    <button id="${system}-play-pause-btn" class="player-btn"${hasAudio ? '' : ' disabled'}>${hasAudio ? PLAY_ICON_SVG : STOP_ICON_SVG}</button>
                    <div class="time-display" id="${system}-current-time">0:00.00</div>
                    <div class="progress-container" id="${system}-progress-container">
                        <div class="progress-bar">
                            <div class="progress-fill" id="${system}-progress-fill"></div>
                        </div>
                    </div>
                    <div class="time-display" id="${system}-total-duration">${totalDurationDisplay}</div>
                    ${downloadActionsHTML}
                </div>
            </div>
        `;
        
        container.insertAdjacentHTML('beforeend', resultHTML);
        container.style.display = 'flex';
        
        if (hasAudio) {
            await setupCustomPlayer(system, result);
        } else {
            const playBtnEl = document.getElementById(`${system}-play-pause-btn`);
            if (playBtnEl) playBtnEl.disabled = true;
        }
    }
    
    // Setup custom audio player for a given system
    async function setupCustomPlayer(system, result) {
        const playPauseBtn = document.getElementById(`${system}-play-pause-btn`);
        const progressContainer = document.getElementById(`${system}-progress-container`);
        const currentTimeDisplay = document.getElementById(`${system}-current-time`);
        const durationDisplay = document.getElementById(`${system}-total-duration`);
        const progressFill = document.getElementById(`${system}-progress-fill`);
        
        if (!playPauseBtn || !progressContainer || !currentTimeDisplay || !durationDisplay || !progressFill) {
            console.error('Failed to find player elements for', system);
            return;
        }
        
        // Create dedicated audio context for this player
        const playerAudioContext = new (window.AudioContext || window.webkitAudioContext)();
        let audioBuffer = null;
        let source = null;
        let startTime = 0;
        let pauseTime = 0;
        let isPlaying = false;
        let isPaused = false;
        let animationFrameId = null;
        let playerRecord = null;
        
        const formatTime = (seconds) => {
            const mins = Math.floor(seconds / 60);
            const secs = seconds % 60;
            const secString = secs.toFixed(2).padStart(5, '0');
            return `${mins}:${secString}`;
        };
        
        // Fetch and decode audio
        try {
            const response = await fetch(result.url);
            const arrayBuffer = await response.arrayBuffer();
            audioBuffer = await playerAudioContext.decodeAudioData(arrayBuffer);
        } catch (error) {
            console.error('Failed to load audio for', system, error);
            playPauseBtn.disabled = true;
            return;
        }
        
        const updateProgress = () => {
            if (!isPlaying || !playerAudioContext) return;
            
            const currentTime = isPaused ? pauseTime : (playerAudioContext.currentTime - startTime);
            const duration = audioBuffer.duration;
            const progress = duration > 0 ? (currentTime / duration) * 100 : 0;
            
            progressFill.style.width = `${Math.min(progress, 100)}%`;
            currentTimeDisplay.textContent = formatTime(Math.min(currentTime, duration), { trimMobile: true });
            
            if (currentTime < duration) {
                animationFrameId = requestAnimationFrame(updateProgress);
            } else {
                // Playback finished
                isPlaying = false;
                isPaused = false;
                playPauseBtn.innerHTML = PLAY_ICON_SVG;
                progressFill.style.width = '100%';
                currentTimeDisplay.textContent = formatTime(duration, { trimMobile: true });
            }
        };
        
        const togglePlayPause = () => {
            if (!audioBuffer) return;
            
            if (isPaused) {
                // Resume from paused position
                pauseAllPlayersExcept(playerRecord);
                if (playerAudioContext.state === 'suspended') {
                    playerAudioContext.resume();
                }
                
                source = playerAudioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(playerAudioContext.destination);
                source.start(0, pauseTime);
                startTime = playerAudioContext.currentTime - pauseTime;
                
                isPaused = false;
                isPlaying = true;
                playPauseBtn.innerHTML = PAUSE_ICON_SVG;
                updateProgress();
            } else if (isPlaying) {
                // Pause playback
                pauseTime = playerAudioContext.currentTime - startTime;
                if (source) {
                    source.stop();
                    source = null;
                }
                playerAudioContext.suspend();
                isPaused = true;
                isPlaying = false;
                playPauseBtn.innerHTML = PLAY_ICON_SVG;
                if (animationFrameId) {
                    cancelAnimationFrame(animationFrameId);
                }
            } else {
                // Start from beginning
                pauseAllPlayersExcept(playerRecord);
                pauseTime = 0;
                
                if (playerAudioContext.state === 'suspended') {
                    playerAudioContext.resume();
                }
                
                source = playerAudioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(playerAudioContext.destination);
                source.start(0);
                startTime = playerAudioContext.currentTime;
                
                isPlaying = true;
                isPaused = false;
                playPauseBtn.innerHTML = PAUSE_ICON_SVG;
                updateProgress();
            }
        };
        
        const seekTo = (percentage) => {
            if (!audioBuffer) return;
            
            const seekTime = (percentage / 100) * audioBuffer.duration;
            const wasPlaying = isPlaying && !isPaused;
            
            // Stop current playback
            if (source) {
                try {
                    source.stop();
                } catch (e) {
                    // Already stopped
                }
                source = null;
            }
            
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
            }
            
            pauseTime = seekTime;
            
            // Update UI
            const progress = (seekTime / audioBuffer.duration) * 100;
            progressFill.style.width = `${Math.min(progress, 100)}%`;
            currentTimeDisplay.textContent = formatTime(seekTime, { trimMobile: true });
            
            if (wasPlaying) {
                // Resume from new position
                if (playerAudioContext.state === 'suspended') {
                    playerAudioContext.resume();
                }
                
                source = playerAudioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(playerAudioContext.destination);
                source.start(0, seekTime);
                startTime = playerAudioContext.currentTime - seekTime;
                
                isPlaying = true;
                isPaused = false;
                playPauseBtn.innerHTML = PAUSE_ICON_SVG;
                updateProgress();
            } else {
                // Just update position, stay paused
                isPaused = true;
                isPlaying = true;
                playPauseBtn.innerHTML = PLAY_ICON_SVG;
            }
        };
        
        // Cleanup function for this player

        const pausePlayback = () => {
            if (!playerAudioContext || playerAudioContext.state === 'closed') return;
            if (isPlaying) {
                pauseTime = playerAudioContext.currentTime - startTime;
                if (source) {
                    try {
                        source.stop();
                    } catch (e) {
                        // Already stopped
                    }
                    source = null;
                }
                playerAudioContext.suspend().catch(() => {});
                isPaused = true;
                isPlaying = false;
                playPauseBtn.innerHTML = PLAY_ICON_SVG;
                if (animationFrameId) {
                    cancelAnimationFrame(animationFrameId);
                    animationFrameId = null;
                }
            }
        };

        const cleanup = () => {
            pausePlayback();
            if (playerAudioContext && playerAudioContext.state !== 'closed') {
                playerAudioContext.close();
            }
            if (playerRecord) {
                customAudioPlayers = customAudioPlayers.filter(p => p !== playerRecord);
            }
        };
        
        playerRecord = {
            audioContext: playerAudioContext,
            cleanup,
            pausePlayback
        };
        
        customAudioPlayers.push(playerRecord);
        
        // Setup event listeners
        playPauseBtn.addEventListener('click', togglePlayPause);
        
        progressContainer.addEventListener('click', (e) => {
            const rect = progressContainer.getBoundingClientRect();
            const percentage = ((e.clientX - rect.left) / rect.width) * 100;
            seekTo(percentage);
        });
    }

    // Generate Supertonic speech with chunking support and progressive playback
    async function generateSupertonicSpeechChunked(text, totalStep, durationFactor, onFirstChunkReady, onChunkAdded) {
        const supertonicStartTime = Date.now();
        const sampleRate = cfgs.ae.sample_rate;
        const silenceDuration = 0.3; // 0.3 seconds of silence between chunks
        
        try {
            // Split text into chunks
            const chunks = chunkText(text);
            
            const audioDataArrays = [];
            const durations = [];
            const silenceSamples = Math.floor(silenceDuration * sampleRate);
            let firstChunkEndTime = 0;
            let firstChunkTime = 0;
            
            // Generate speech for each chunk
            for (let i = 0; i < chunks.length; i++) {
                const chunkText = chunks[i];
                
                const result = await generateSupertonicSpeech(chunkText, totalStep, durationFactor);
                
                if (!result.success) {
                    throw new Error(`Failed to generate chunk ${i + 1}: ${result.error}`);
                }
                
                // Fetch and parse the WAV file using the existing parseWavFile function
                const response = await fetch(result.url);
                const arrayBuffer = await response.arrayBuffer();
                const { audioData } = parseWavFile(arrayBuffer);
                
                audioDataArrays.push(audioData);
                durations.push(result.audioDuration);
                
                // Clean up the blob URL
                URL.revokeObjectURL(result.url);
                
                // Progressive playback: send each chunk individually for Web Audio API
                if (i === 0 && onFirstChunkReady) {
                    // First chunk ready - send it immediately
                    firstChunkEndTime = Date.now();
                    firstChunkTime = (firstChunkEndTime - supertonicStartTime) / 1000;
                    
                    const initialWav = writeWavFile(audioData, sampleRate);
                    const initialBlob = new Blob([initialWav], { type: 'audio/wav' });
                    const initialUrl = URL.createObjectURL(initialBlob);
                    const totalDurationSoFar = result.audioDuration;
                    const processedChars = chunks[0].length;
                    onFirstChunkReady(initialUrl, totalDurationSoFar, text, chunks.length, firstChunkTime, processedChars);
                } else if (i > 0 && onChunkAdded) {
                    // Subsequent chunks - send just the new chunk
                    const chunkWav = writeWavFile(audioData, sampleRate);
                    const chunkBlob = new Blob([chunkWav], { type: 'audio/wav' });
                    const chunkUrl = URL.createObjectURL(chunkBlob);
                    
                    const totalDurationSoFar = durations.slice(0, i + 1).reduce((sum, dur) => sum + dur, 0) + silenceDuration * i;
                    const currentProcessingTime = (Date.now() - supertonicStartTime) / 1000;
                    const processedChars = chunks.slice(0, i + 1).reduce((sum, chunk) => sum + chunk.length, 0);
                    onChunkAdded(chunkUrl, totalDurationSoFar, i + 1, chunks.length, currentProcessingTime, processedChars);
                }
            }
            
            // Concatenate all audio chunks with silence for final result
            const totalDuration = durations.reduce((sum, dur) => sum + dur, 0) + silenceDuration * (chunks.length - 1);
            
            // Calculate total samples needed
            let totalSamples = 0;
            for (let i = 0; i < audioDataArrays.length; i++) {
                totalSamples += audioDataArrays[i].length;
                if (i < audioDataArrays.length - 1) {
                    totalSamples += silenceSamples;
                }
            }
            
            const wavCat = new Float32Array(totalSamples);
            
            let currentIdx = 0;
            for (let i = 0; i < audioDataArrays.length; i++) {
                // Copy audio data
                const audioData = audioDataArrays[i];
                wavCat.set(audioData, currentIdx);
                currentIdx += audioData.length;
                
                // Add silence if not the last chunk
                if (i < audioDataArrays.length - 1) {
                    // Silence is already zeros in Float32Array, just skip the indices
                    currentIdx += silenceSamples;
                }
            }
            
            // Create final WAV file
            const wavBuffer = writeWavFile(wavCat, sampleRate);
            const blob = new Blob([wavBuffer], { type: 'audio/wav' });
            const url = URL.createObjectURL(blob);
            
            const supertonicEndTime = Date.now();
            const supertonicProcessingTime = (supertonicEndTime - supertonicStartTime) / 1000;
            
            return {
                success: true,
                processingTime: supertonicProcessingTime,
                audioDuration: totalDuration,
                url: url,
                text: text,
                firstChunkTime: firstChunkTime
            };
        } catch (error) {
            return {
                success: false,
                error: error.message,
                text: text
            };
        }
    }

    // Main synthesis function
    async function generateSpeech() {
        const text = demoTextInput.value.trim();
        
        // Validate text input
        const validation = validateTextInput(text);
        if (!validation.valid) {
            showDemoError(validation.message);
            return;
        }
        
        if (!models || !cfgs || !processors) {
            showDemoError('Models are still loading. Please wait.');
            return;
        }
        
        if (!currentStyleTtlTensor || !currentStyleDpTensor) {
            showDemoError('Reference embeddings are not ready. Please wait.');
            return;
        }
        
        // Validate characters before generation
        const charValidation = validateCharacters(text);
        if (!charValidation.valid && charValidation.unsupportedChars.length > 0) {
            const charList = charValidation.unsupportedChars.map(c => `"${c}"`).join(', ');
            showDemoError(`Cannot generate speech: Unsupported characters found: ${charList}`);
            return;
        }
        
        const elevenlabsApiKey = demoElevenLabsApiKey.value.trim();
        const openaiApiKey = demoSecondaryApiKey.value.trim();
        const geminiApiKey = demoTertiaryApiKey.value.trim();
        const hasComparison = !!elevenlabsApiKey || !!openaiApiKey || !!geminiApiKey;
        isComparisonMode = hasComparison;
        document.body.classList.toggle('comparison-mode', hasComparison);

        currentGenerationTextLength = text.length;
        
        // Show billing confirmation if API keys are provided
        if (hasComparison) {
            const apiProviders = [];
            if (elevenlabsApiKey) apiProviders.push('ElevenLabs Flash v2.5');
            if (openaiApiKey) apiProviders.push('OpenAI TTS-1');
            if (geminiApiKey) apiProviders.push('Gemini 2.5 Flash TTS');
            
            const userConfirmed = await showBillingConfirmation(text.length, apiProviders);
            
            if (!userConfirmed) {
                return;
            }
        }
        
        if (!hasComparison && demoComparisonSection) {
            demoComparisonSection.style.display = 'none';
        }
        
        try {
            isGenerating = true;
            demoGenerateBtn.disabled = true;
            
            // Disable voice toggle during generation
            const voiceToggleTexts = document.querySelectorAll('.voice-toggle-text');
            voiceToggleTexts.forEach(text => text.classList.add('disabled'));
            
            hideDemoError();
            hideDemoStatus(); // Hide the status box when starting generation
            
            // Clean up previous audio playback
            if (audioContext) {
                // Stop all scheduled sources
                scheduledSources.forEach(source => {
                    try {
                        source.stop();
                    } catch (e) {
                        // Already stopped
                    }
                });
                scheduledSources = [];
                
                // Close audio context
                if (audioContext.state !== 'closed') {
                    audioContext.close();
                }
                audioContext = null;
            }
            
            // Cancel animation frame
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
                animationFrameId = null;
            }
            
            // Clean up all custom audio players (ElevenLabs, etc.)
            customAudioPlayers.forEach(player => {
                if (player.cleanup) {
                    player.cleanup();
                }
            });
            customAudioPlayers = [];
            
            // Reset state
            audioChunks = [];
            totalDuration = 0;
            startTime = 0;
            pauseTime = 0;
            isPaused = false;
            isPlaying = false;
            firstChunkGenerationTime = 0; // Processing time for first chunk
            totalChunks = 0;
            nextScheduledTime = 0; // Next time to schedule audio chunk
            
            // Show result shell(s) immediately
            const createInitialResultItem = (system, titleMain, titleSub, titleColor, includeStatus) => {
                const titleStatus = includeStatus
                    ? `<span class="title-status status-running" id="${system}-status">⏳ Running...</span>`
                    : '';
                return `
                    <div class="demo-result-item ${system}-result-item generating" id="${system}-result" style="--result-progress: 0%;">
                        <div class="demo-result-title">
                            <span class="title-main" style="color: ${titleColor};">${titleMain}</span>
                            <span class="title-sub">${titleSub}</span>
                            ${titleStatus}
                        </div>
                        <div class="demo-result-info">
                    <!--
                    <div class="stat">
                        <div class="stat-value" id="${system}-chars">--</div>
                        <div class="stat-label">Processed Chars</div>
                    </div>
                    -->
                            <div class="stat">
                                <div class="stat-value" id="${system}-time">--</div>
                            <div class="stat-label">Processing Time<span class="stat-arrow stat-arrow--down">↓</span></div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="${system}-cps">--</div>
                            <div class="stat-label">Chars/sec<span class="stat-arrow stat-arrow--up">↑</span></div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="${system}-rtf">--</div>
                            <div class="stat-label">RTF<span class="stat-arrow stat-arrow--down">↓</span></div>
                            </div>
                        </div>
                        <div class="custom-audio-player">
                        <div class="demo-placeholder-audio">Generating speech...</div>
                        </div>
                    </div>
                `;
            };
            const supertonicInitial = createInitialResultItem(
                'supertonic',
                'Supertonic',
                'On-Device',
                'var(--supertone_blue)',
                isComparisonMode
            );
            const initialItems = [supertonicInitial];
            if (elevenlabsApiKey) {
                const elevenInitial = createInitialResultItem(
                    'elevenlabs',
                    'ElevenLabs Flash v2.5',
                    'Cloud API',
                    '#999',
                    true
                );
                initialItems.push(elevenInitial);
            }
            if (openaiApiKey) {
                const openaiInitial = createInitialResultItem(
                    'openai',
                    'OpenAI TTS-1',
                    'Cloud API',
                    '#999',
                    true
                );
                initialItems.push(openaiInitial);
            }
            if (geminiApiKey) {
                const geminiInitial = createInitialResultItem(
                    'gemini',
                    'Gemini 2.5 Flash TTS',
                    'Cloud API',
                    '#999',
                    true
                );
                initialItems.push(geminiInitial);
            }
            demoResults.style.display = 'flex';
            demoResults.innerHTML = initialItems.join('');
            
            // Reset comparison table
            if (hasComparison) {
                demoComparisonSection.style.display = 'block';
                document.getElementById('supertonicStatus').textContent = '⏳ Running...';
                document.getElementById('supertonicStatus').className = 'demo-comparison-cell status-running';
                document.getElementById('supertonicTime').textContent = '-';
                document.getElementById('supertonicDuration').textContent = '-';
                document.getElementById('supertonicRTF').textContent = '-';
                
                if (elevenlabsApiKey) {
                    document.getElementById('elevenlabsStatus').textContent = '⏳ Running...';
                    document.getElementById('elevenlabsStatus').className = 'demo-comparison-cell status-running';
                    document.getElementById('elevenlabsTime').textContent = '-';
                    document.getElementById('elevenlabsDuration').textContent = '-';
                    document.getElementById('elevenlabsRTF').textContent = '-';
                }
                
                if (openaiApiKey) {
                    document.getElementById('openaiStatus').textContent = '⏳ Running...';
                    document.getElementById('openaiStatus').className = 'demo-comparison-cell status-running';
                    document.getElementById('openaiTime').textContent = '-';
                    document.getElementById('openaiDuration').textContent = '-';
                    document.getElementById('openaiRTF').textContent = '-';
                }
                
                if (geminiApiKey) {
                    document.getElementById('geminiStatus').textContent = '⏳ Running...';
                    document.getElementById('geminiStatus').className = 'demo-comparison-cell status-running';
                    document.getElementById('geminiTime').textContent = '-';
                    document.getElementById('geminiDuration').textContent = '-';
                    document.getElementById('geminiRTF').textContent = '-';
                }
                
                // Remove winner classes
                document.querySelector('.supertonic-row').classList.remove('winner');
                const elevenlabsRow = document.querySelector('.elevenlabs-row');
                const openaiRow = document.querySelector('.openai-row');
                const geminiRow = document.querySelector('.gemini-row');
                if (elevenlabsRow) elevenlabsRow.classList.remove('winner');
                if (openaiRow) openaiRow.classList.remove('winner');
                if (geminiRow) geminiRow.classList.remove('winner');
            }
            
            const totalStep = parseInt(demoTotalSteps.value);
            const durationFactor = parseFloat(demoDurationFactor.value);
            
            // Track which one finishes first
            let firstFinished = false;
            let supertonicResult = null;
            let elevenlabsResult = null;
            let openaiResult = null;
            let geminiResult = null;
            let latestSupertonicProcessedChars = 0;
            
            // Helper functions for custom player
            const formatTime = (seconds, { trimMobile = false } = {}) => {
                const mins = Math.floor(seconds / 60);
                const secs = seconds % 60;
                const secString = secs.toFixed(2).padStart(5, '0');
                let formatted = `${mins}:${secString}`;
                if (trimMobile) {
                    formatted = trimDecimalsForMobile(formatted);
                }
                return formatted;
            };
            
            const updateProgress = () => {
                if (!isPlaying || !audioContext) return;
                
                const currentTime = isPaused ? pauseTime : (audioContext.currentTime - startTime);
                const progress = totalDuration > 0 ? (currentTime / totalDuration) * 100 : 0;
                
                if (progressFill) {
                    progressFill.style.width = `${Math.min(progress, 100)}%`;
                }
                if (currentTimeDisplay) {
                    currentTimeDisplay.textContent = formatTime(Math.min(currentTime, totalDuration), { trimMobile: true });
                }
                
                if (currentTime < totalDuration) {
                    animationFrameId = requestAnimationFrame(updateProgress);
                } else {
                    // Playback finished
                    isPlaying = false;
                    isPaused = false;
                    if (playPauseBtn) {
                        playPauseBtn.innerHTML = PLAY_ICON_SVG;
                    }
                }
            };
            
            const togglePlayPause = () => {
                if (!audioContext || audioChunks.length === 0) return;
                
                if (isPaused) {
                    // Resume from paused position
                    pauseAllPlayersExcept(supertonicPlayerRecord);
                    
                    const seekTime = pauseTime;
                    
                    // Find which chunk we should start from
                    let accumulatedTime = 0;
                    let startChunkIndex = 0;
                    let offsetInChunk = seekTime;
                    
                    for (let i = 0; i < audioChunks.length; i++) {
                        const chunkDuration = audioChunks[i].buffer.duration;
                        if (accumulatedTime + chunkDuration > seekTime) {
                            startChunkIndex = i;
                            offsetInChunk = seekTime - accumulatedTime;
                            break;
                        }
                        accumulatedTime += chunkDuration + 0.3;
                    }
                    
                    // Stop any existing sources
                    scheduledSources.forEach(source => {
                        try {
                            source.stop();
                        } catch (e) {
                            // Already stopped
                        }
                    });
                    scheduledSources = [];
                    
                    // Resume AudioContext if suspended
                    if (audioContext.state === 'suspended') {
                        audioContext.resume();
                    }
                    
                    // Reschedule from the pause point
                    startTime = audioContext.currentTime - seekTime;
                    let nextStartTime = audioContext.currentTime;
                    
                    for (let i = startChunkIndex; i < audioChunks.length; i++) {
                        const source = audioContext.createBufferSource();
                        source.buffer = audioChunks[i].buffer;
                        source.connect(audioContext.destination);
                        
                        if (i === startChunkIndex) {
                            source.start(nextStartTime, offsetInChunk);
                            nextStartTime += (audioChunks[i].buffer.duration - offsetInChunk);
                        } else {
                            source.start(nextStartTime);
                            nextStartTime += audioChunks[i].buffer.duration;
                        }
                        
                        if (i < audioChunks.length - 1) {
                            nextStartTime += 0.3;
                        }
                        
                        scheduledSources.push(source);
                    }
                    
                    nextScheduledTime = nextStartTime;
                    
                    isPaused = false;
                    isPlaying = true;
                    playPauseBtn.innerHTML = PAUSE_ICON_SVG;
                    updateProgress();
                } else if (isPlaying) {
                    // Pause playback
                    pauseTime = audioContext.currentTime - startTime;
                    audioContext.suspend();
                    isPaused = true;
                    playPauseBtn.innerHTML = PLAY_ICON_SVG;
                    if (animationFrameId) {
                        cancelAnimationFrame(animationFrameId);
                    }
                } else {
                    // Was finished, restart from beginning
                    pauseAllPlayersExcept(supertonicPlayerRecord);
                    
                    pauseTime = 0;
                    
                    // Resume AudioContext if suspended
                    if (audioContext.state === 'suspended') {
                        audioContext.resume();
                    }
                    
                    // Stop any existing sources
                    scheduledSources.forEach(source => {
                        try {
                            source.stop();
                        } catch (e) {
                            // Already stopped
                        }
                    });
                    scheduledSources = [];
                    
                    // Restart from beginning
                    startTime = audioContext.currentTime;
                    let nextStartTime = audioContext.currentTime;
                    
                    for (let i = 0; i < audioChunks.length; i++) {
                        const source = audioContext.createBufferSource();
                        source.buffer = audioChunks[i].buffer;
                        source.connect(audioContext.destination);
                        source.start(nextStartTime);
                        nextStartTime += audioChunks[i].buffer.duration;
                        
                        if (i < audioChunks.length - 1) {
                            nextStartTime += 0.3;
                        }
                        
                        scheduledSources.push(source);
                    }
                    
                    nextScheduledTime = nextStartTime;
                    
                    isPlaying = true;
                    isPaused = false;
                    playPauseBtn.innerHTML = PAUSE_ICON_SVG;
                    updateProgress();
                }
            };
            
            const seekTo = (percentage) => {
                if (!audioContext || audioChunks.length === 0) return;
                
                const seekTime = (percentage / 100) * totalDuration;
                
                // Remember current playing state
                const wasPlaying = isPlaying;
                const wasPaused = isPaused;
                
                // Stop all current sources
                scheduledSources.forEach(source => {
                    try {
                        source.stop();
                    } catch (e) {
                        // Already stopped
                    }
                });
                scheduledSources = [];
                
                // Cancel animation
                if (animationFrameId) {
                    cancelAnimationFrame(animationFrameId);
                }
                
                // Find which chunk we should start from
                let accumulatedTime = 0;
                let startChunkIndex = 0;
                let offsetInChunk = seekTime;
                
                for (let i = 0; i < audioChunks.length; i++) {
                    const chunkDuration = audioChunks[i].buffer.duration;
                    if (accumulatedTime + chunkDuration > seekTime) {
                        startChunkIndex = i;
                        offsetInChunk = seekTime - accumulatedTime;
                        break;
                    }
                    accumulatedTime += chunkDuration + 0.3; // Include silence
                }
                
                // If paused or finished, just update the pause position
                if (wasPaused || !wasPlaying) {
                    pauseTime = seekTime;
                    
                    // Update UI
                    if (progressFill) {
                        const progress = (seekTime / totalDuration) * 100;
                        progressFill.style.width = `${Math.min(progress, 100)}%`;
                    }
                    if (currentTimeDisplay) {
                    currentTimeDisplay.textContent = formatTime(seekTime, { trimMobile: true });
                    }
                    
                    // Set to paused state so play button will resume from seek position
                    isPaused = true;
                    isPlaying = true; // Valid state for playback
                    
                    if (playPauseBtn) {
                        playPauseBtn.innerHTML = PLAY_ICON_SVG;
                    }
                    
                    return;
                }
                
                // Resume AudioContext if it was suspended
                if (audioContext.state === 'suspended') {
                    audioContext.resume();
                }
                
                // Reschedule from the seek point
                startTime = audioContext.currentTime - seekTime;
                let nextStartTime = audioContext.currentTime;
                
                for (let i = startChunkIndex; i < audioChunks.length; i++) {
                    const source = audioContext.createBufferSource();
                    source.buffer = audioChunks[i].buffer;
                    source.connect(audioContext.destination);
                    
                    if (i === startChunkIndex) {
                        // Start from offset
                        source.start(nextStartTime, offsetInChunk);
                        nextStartTime += (audioChunks[i].buffer.duration - offsetInChunk);
                    } else {
                        source.start(nextStartTime);
                        nextStartTime += audioChunks[i].buffer.duration;
                    }
                    
                    // Add silence between chunks
                    if (i < audioChunks.length - 1) {
                        nextStartTime += 0.3;
                    }
                    
                    scheduledSources.push(source);
                }
                
                // Update nextScheduledTime for any future chunks
                nextScheduledTime = nextStartTime;
                
                // Resume playing state
                isPlaying = true;
                isPaused = false;
                if (playPauseBtn) {
                    playPauseBtn.innerHTML = PAUSE_ICON_SVG;
                }
                
                // Restart progress animation
                updateProgress();
            };
            
            // Callback for first chunk ready - create custom player and start playback
            const onFirstChunkReady = async (url, duration, text, numChunks, firstChunkTime, processedChars) => {
                totalChunks = numChunks;
                firstChunkGenerationTime = firstChunkTime;
                
                const container = document.getElementById('demoResults');
                
                if (!firstFinished) {
                    firstFinished = true;
                }

                const textLength = currentGenerationTextLength > 0
                    ? currentGenerationTextLength
                    : (text ? text.length : 0);
                const isBatch = textLength >= MAX_CHUNK_LENGTH;
                const processingTimeStr = isBatch && firstChunkTime
                    ? `${formatTimeDetailed(firstChunkTime)} / ${formatTimeDetailed(firstChunkTime)}`
                    : formatTimeDetailed(firstChunkTime);
                const safeInitialChars = typeof processedChars === 'number' ? processedChars : 0;
                const displayedInitialChars = textLength > 0 ? Math.min(safeInitialChars, textLength) : safeInitialChars;
                const charsPerSec = firstChunkTime > 0 && displayedInitialChars > 0
                    ? (displayedInitialChars / firstChunkTime).toFixed(1)
                    : '0.0';
                const rtf = duration > 0 && firstChunkTime > 0 ? (firstChunkTime / duration).toFixed(3) : '-';
                const progressValue = textLength > 0 ? Math.min(100, (displayedInitialChars / textLength) * 100) : 0;

                const resultItemEl = document.getElementById('supertonic-result');
                if (!resultItemEl) {
                    console.warn('Supertonic result container not found.');
                    return;
                }

                resultItemEl.classList.remove('generating');
                resultItemEl.style.setProperty('--result-progress', `${progressValue}%`);

                const titleMainEl = resultItemEl.querySelector('.title-main');
                if (titleMainEl) {
                    titleMainEl.textContent = 'Supertonic';
                    titleMainEl.style.color = 'var(--supertone_blue)';
                }
                const titleSubEl = resultItemEl.querySelector('.title-sub');
                if (titleSubEl) {
                    titleSubEl.textContent = 'On-Device';
                }

                const infoContainer = resultItemEl.querySelector('.demo-result-info');
                if (infoContainer) {
                    infoContainer.classList.remove('error');
                }
                const timeElInitial = document.getElementById('supertonic-time');
                if (timeElInitial) {
                    timeElInitial.innerHTML = formatStatValueWithSuffix(processingTimeStr, 's', { firstLabel: true });
                }
                const cpsElInitial = document.getElementById('supertonic-cps');
                if (cpsElInitial) {
                    cpsElInitial.textContent = charsPerSec;
                }
                const rtfElInitial = document.getElementById('supertonic-rtf');
                if (rtfElInitial) {
                    rtfElInitial.innerHTML = formatStatValueWithSuffix(rtf, 'x');
                }

                const playerContainer = resultItemEl.querySelector('.custom-audio-player');
                if (playerContainer) {
                    playerContainer.style.display = '';
                    playerContainer.innerHTML = `
                        <button id="play-pause-btn" class="player-btn">${PAUSE_ICON_SVG}</button>
                        <div class="time-display" id="current-time">0:00.00</div>
                        <div class="progress-container" id="progress-container">
                            <div class="progress-bar">
                                <div class="progress-fill" id="progress-fill"></div>
                            </div>
                        </div>
                        <div class="time-display" id="total-duration">${formatTime(duration, { trimMobile: true })}</div>
                        <div class="demo-result-actions" style="display: none;">
                            <button class="demo-download-btn" id="supertonic-download" aria-label="Download WAV" title="Download WAV">
                                <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                                    <polyline points="7 10 12 15 17 10"/>
                                    <line x1="12" y1="15" x2="12" y2="3"/>
                                </svg>
                            </button>
                        </div>
                    `;
                }

                container.style.display = 'flex';
                latestSupertonicProcessedChars = displayedInitialChars;
                
                 // Get UI elements
                 playPauseBtn = document.getElementById('play-pause-btn');
                 progressBar = document.getElementById('progress-container');
                 currentTimeDisplay = document.getElementById('current-time');
                durationDisplay = document.getElementById('total-duration');
                progressFill = document.getElementById('progress-fill');
                
                // Initialize Web Audio API
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                startTime = audioContext.currentTime;
                totalDuration = duration;
                isPlaying = true;
                isPaused = false;
                
                // Create Supertonic player record and register it
                const pausePlayback = () => {
                    if (!audioContext || audioContext.state === 'closed') return;
                    if (isPlaying) {
                        pauseTime = audioContext.currentTime - startTime;
                        scheduledSources.forEach(source => {
                            try {
                                source.stop();
                            } catch (e) {
                                // Already stopped
                            }
                        });
                        scheduledSources = [];
                        audioContext.suspend();
                        isPaused = true;
                        isPlaying = false;
                        if (playPauseBtn) {
                            playPauseBtn.innerHTML = PLAY_ICON_SVG;
                        }
                        if (animationFrameId) {
                            cancelAnimationFrame(animationFrameId);
                        }
                    }
                };
                
                supertonicPlayerRecord = {
                    audioContext: audioContext,
                    pausePlayback: pausePlayback
                };
                
                // Remove old Supertonic player if exists and add new one
                customAudioPlayers = customAudioPlayers.filter(p => p !== supertonicPlayerRecord && p.audioContext !== audioContext);
                customAudioPlayers.push(supertonicPlayerRecord);
                
                // Pause all other players before starting Supertonic
                pauseAllPlayersExcept(supertonicPlayerRecord);
                
                // Fetch and decode first chunk
                const response = await fetch(url);
                const arrayBuffer = await response.arrayBuffer();
                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                
                audioChunks.push({ buffer: audioBuffer, duration: audioBuffer.duration });
                
                // Play first chunk immediately
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContext.destination);
                source.start(audioContext.currentTime);
                scheduledSources.push(source);
                
                // Set next scheduled time for additional chunks
                nextScheduledTime = audioContext.currentTime + audioBuffer.duration + 0.3; // Add silence gap
                
                // Setup player controls
                playPauseBtn.addEventListener('click', togglePlayPause);
                
                progressBar.addEventListener('click', (e) => {
                    const rect = progressBar.getBoundingClientRect();
                    const percentage = ((e.clientX - rect.left) / rect.width) * 100;
                    seekTo(percentage);
                });
                
                // Start progress animation
                updateProgress();
                
                // Clean up URL
                URL.revokeObjectURL(url);
            };
            
            // Callback for each additional chunk - schedule seamlessly
            const onChunkAdded = async (url, duration, chunkIndex, totalChunks, currentProcessingTime, processedChars) => {
                if (!audioContext) return;
                
                // Fetch and decode the new chunk
                const response = await fetch(url);
                const arrayBuffer = await response.arrayBuffer();
                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                
                const chunkDuration = audioBuffer.duration;
                audioChunks.push({ buffer: audioBuffer, duration: chunkDuration });
                
                // Schedule the new chunk at the pre-calculated time
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContext.destination);
                source.start(nextScheduledTime);
                scheduledSources.push(source);
                
                // Update next scheduled time for the next chunk
                nextScheduledTime = nextScheduledTime + audioBuffer.duration + 0.3; // Add silence gap
                
                // Update total duration
                totalDuration = duration;
                
                // Update duration display with smooth animation
                if (durationDisplay) {
                    durationDisplay.textContent = formatTime(duration, { trimMobile: true });
                    durationDisplay.style.transition = 'color 0.3s';
                    durationDisplay.style.color = 'var(--supertone_blue)';
                    setTimeout(() => {
                        durationDisplay.style.color = '';
                    }, 300);
                }
                
                // Update info display
                const textLengthCandidate = currentGenerationTextLength > 0
                    ? currentGenerationTextLength
                    : demoTextInput.value.trim().length;
                const textLength = textLengthCandidate;
                const isBatch = textLength >= MAX_CHUNK_LENGTH;
                const timeEl = document.getElementById('supertonic-time');
                const durationEl = document.getElementById('supertonic-duration');
                const cpsEl = document.getElementById('supertonic-cps');
                const rtfEl = document.getElementById('supertonic-rtf');
                const effectiveProcessedChars = typeof processedChars === 'number' ? processedChars : latestSupertonicProcessedChars;

                if (effectiveProcessedChars < latestSupertonicProcessedChars) {
                    URL.revokeObjectURL(url);
                    return;
                }

                const clampedProcessedChars = textLength > 0 ? Math.min(effectiveProcessedChars, textLength) : effectiveProcessedChars;
                const progressValue = textLength > 0 ? Math.min(100, (clampedProcessedChars / textLength) * 100) : 0;
                if (durationEl) {
                    durationEl.textContent = formatTimeDetailed(duration);
                }
                if (timeEl && isBatch && firstChunkGenerationTime > 0 && currentProcessingTime) {
                    const timeDisplay = `${formatTimeDetailed(firstChunkGenerationTime)} / ${formatTimeDetailed(currentProcessingTime)}`;
                    timeEl.innerHTML = formatStatValueWithSuffix(timeDisplay, 's', { firstLabel: true });
                }
                if (cpsEl && currentProcessingTime > 0 && clampedProcessedChars >= 0) {
                    const charsPerSec = (clampedProcessedChars / currentProcessingTime).toFixed(1);
                    cpsEl.textContent = charsPerSec;
                }
                if (rtfEl && duration > 0 && currentProcessingTime > 0) {
                    const rtf = (currentProcessingTime / duration).toFixed(3);
                    rtfEl.innerHTML = formatStatValueWithSuffix(rtf, 'x');
                }
                const resultItemEl = document.getElementById('supertonic-result');
                if (resultItemEl) {
                    resultItemEl.style.setProperty('--result-progress', `${progressValue}%`);
                }
                latestSupertonicProcessedChars = clampedProcessedChars;
                
                // Clean up URL
                URL.revokeObjectURL(url);
            };
            
            // Start all syntheses simultaneously
            const supertonicPromise = generateSupertonicSpeechChunked(
                text, 
                totalStep, 
                durationFactor, 
                onFirstChunkReady, 
                onChunkAdded
            );
            const elevenlabsPromise = elevenlabsApiKey ? generateSpeechElevenLabs(text, elevenlabsApiKey) : null;
            const openaiPromise = openaiApiKey ? generateSpeechOpenAI(text, openaiApiKey) : null;
            const geminiPromise = geminiApiKey ? generateSpeechGemini(text, geminiApiKey) : null;
            
            // Handle results as they arrive
                supertonicPromise.then(result => {
                supertonicResult = result;
                if (result.success) {
                    const textLength = result.text ? result.text.length : 0;
                    const isBatch = textLength >= MAX_CHUNK_LENGTH;
                    const processingTimeStr = isBatch && firstChunkGenerationTime > 0
                        ? `${formatTimeDetailed(firstChunkGenerationTime)} / ${formatTimeDetailed(result.processingTime)}`
                        : formatTimeDetailed(result.processingTime);
                    const charsPerSec = result.processingTime > 0 ? (textLength / result.processingTime).toFixed(1) : '0.0';
                    const progressValue = textLength > 0 ? 100 : 0;
                    const progressDisplay = progressValue.toFixed(1);
                    
                    const timeEl = document.getElementById('supertonic-time');
                    const durationEl = document.getElementById('supertonic-duration');
                    const cpsEl = document.getElementById('supertonic-cps');
                    const rtfEl = document.getElementById('supertonic-rtf');
                    
                    if (timeEl) timeEl.innerHTML = formatStatValueWithSuffix(processingTimeStr, 's', { firstLabel: true });
                    if (durationEl) durationEl.textContent = formatTimeDetailed(result.audioDuration);
                    latestSupertonicProcessedChars = textLength;
                    if (cpsEl) cpsEl.textContent = charsPerSec;
                    if (rtfEl) {
                        const rtf = result.audioDuration > 0 ? (result.processingTime / result.audioDuration).toFixed(3) : '-';
                        rtfEl.innerHTML = formatStatValueWithSuffix(rtf, 'x');
                    }
                    const resultItemEl = document.getElementById('supertonic-result');
                    if (resultItemEl) {
                        resultItemEl.style.setProperty('--result-progress', `${progressValue}%`);
                    }
                    latestSupertonicProcessedChars = textLength;
                    
                    // Final duration update (if custom player was used)
                    if (audioContext && audioChunks.length > 0) {
                        totalDuration = result.audioDuration;
                        if (durationDisplay) {
                            durationDisplay.textContent = formatTime(result.audioDuration, { trimMobile: true });
                        }
                    }
                    
                    // Always show download button
                    const downloadBtn = document.getElementById('supertonic-download');
                    if (downloadBtn) {
                        downloadBtn.parentElement.style.display = 'block';
                        downloadBtn.onclick = () => downloadDemoAudio(result.url, 'supertonic_speech.wav');
                    }
                }
                // Update comparison table immediately
                if (hasComparison) {
                    updateComparisonRow('supertonic', result);
                    // Highlight winner if all are done
                    const allResults = [supertonicResult, elevenlabsResult, openaiResult, geminiResult].filter(r => r !== null);
                    const allFinished = (!elevenlabsApiKey || elevenlabsResult) && (!openaiApiKey || openaiResult) && (!geminiApiKey || geminiResult);
                    if (allFinished && allResults.length > 1) {
                        highlightWinner(allResults);
                    }
                }
            });
            
            if (elevenlabsPromise) {
                elevenlabsPromise.then(result => {
                    elevenlabsResult = result;
                    renderResult('elevenlabs', result, !firstFinished);
                    if (!firstFinished) firstFinished = true;
                    // Update comparison table immediately
                    updateComparisonRow('elevenlabs', result);
                    // Highlight winner if all are done
                    const allResults = [supertonicResult, elevenlabsResult, openaiResult, geminiResult].filter(r => r !== null);
                    const allFinished = (!elevenlabsApiKey || elevenlabsResult) && (!openaiApiKey || openaiResult) && (!geminiApiKey || geminiResult);
                    if (allFinished && allResults.length > 1) {
                        highlightWinner(allResults);
                    }
                });
            }
            
            if (openaiPromise) {
                openaiPromise.then(result => {
                    openaiResult = result;
                    renderResult('openai', result, !firstFinished);
                    if (!firstFinished) firstFinished = true;
                    // Update comparison table immediately
                    updateComparisonRow('openai', result);
                    // Highlight winner if all are done
                    const allResults = [supertonicResult, elevenlabsResult, openaiResult, geminiResult].filter(r => r !== null);
                    const allFinished = (!elevenlabsApiKey || elevenlabsResult) && (!openaiApiKey || openaiResult) && (!geminiApiKey || geminiResult);
                    if (allFinished && allResults.length > 1) {
                        highlightWinner(allResults);
                    }
                });
            }
            
            if (geminiPromise) {
                geminiPromise.then(result => {
                    geminiResult = result;
                    renderResult('gemini', result, !firstFinished);
                    if (!firstFinished) firstFinished = true;
                    // Update comparison table immediately
                    updateComparisonRow('gemini', result);
                    // Highlight winner if all are done
                    const allResults = [supertonicResult, elevenlabsResult, openaiResult, geminiResult].filter(r => r !== null);
                    const allFinished = (!elevenlabsApiKey || elevenlabsResult) && (!openaiApiKey || openaiResult) && (!geminiApiKey || geminiResult);
                    if (allFinished && allResults.length > 1) {
                        highlightWinner(allResults);
                    }
                });
            }
            
            // Wait for all to complete
            await Promise.allSettled([supertonicPromise, elevenlabsPromise, openaiPromise, geminiPromise].filter(p => p !== null));
            
            // If no API key, mark as skipped
            if (!elevenlabsApiKey && hasComparison) {
                const elevenlabsStatus = document.getElementById('elevenlabsStatus');
                const elevenlabsTime = document.getElementById('elevenlabsTime');
                if (elevenlabsStatus) {
                    elevenlabsStatus.textContent = '⏭️ Skipped';
                    elevenlabsStatus.className = 'demo-comparison-cell';
                }
                if (elevenlabsTime) {
                    elevenlabsTime.textContent = 'No API key';
                }
            }
            
            if (!openaiApiKey && hasComparison) {
                const openaiStatus = document.getElementById('openaiStatus');
                const openaiTime = document.getElementById('openaiTime');
                if (openaiStatus) {
                    openaiStatus.textContent = '⏭️ Skipped';
                    openaiStatus.className = 'demo-comparison-cell';
                }
                if (openaiTime) {
                    openaiTime.textContent = 'No API key';
                }
            }
            
            if (!geminiApiKey && hasComparison) {
                const geminiStatus = document.getElementById('geminiStatus');
                const geminiTime = document.getElementById('geminiTime');
                if (geminiStatus) {
                    geminiStatus.textContent = '⏭️ Skipped';
                    geminiStatus.className = 'demo-comparison-cell';
                }
                if (geminiTime) {
                    geminiTime.textContent = 'No API key';
                }
            }
            
        } catch (error) {
            showDemoStatus(`<strong>Error:</strong> ${error.message}`, 'error');
            showDemoError(`Error during synthesis: ${error.message}`);
            console.error('Synthesis error:', error);
            
            // Restore placeholder
            demoResults.style.display = 'none';
            demoResults.innerHTML = `
                <div class="demo-placeholder">
                    <div class="demo-placeholder-icon">🎙️</div>
                    <p>Your generated speech will appear here</p>
                </div>
            `;
        } finally {
            isGenerating = false;
            demoGenerateBtn.disabled = false;
            
            // Re-enable voice toggle after generation
            const voiceToggleTexts = document.querySelectorAll('.voice-toggle-text');
            voiceToggleTexts.forEach(text => text.classList.remove('disabled'));
        }
    }

    // Download handler (make it global)
    window.downloadDemoAudio = function(url, filename) {
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
    };

    // Update slider value displays
    function updateSliderValues() {
        demoTotalStepsValue.textContent = demoTotalSteps.value;
        // Remove unnecessary trailing zeros (1.00 -> 1, 0.80 -> 0.8, 0.75 -> 0.75)
        demoDurationFactorValue.textContent = parseFloat(parseFloat(demoDurationFactor.value).toFixed(2));
    }
    
    // Attach slider event listeners
    demoTotalSteps.addEventListener('input', updateSliderValues);
    demoDurationFactor.addEventListener('input', updateSliderValues);
    
    // Initialize slider values
    updateSliderValues();

    // Attach generate function to button
    demoGenerateBtn.addEventListener('click', generateSpeech);

    // Preset text buttons (defined before input listener to share scope)
    const presetButtons = document.querySelectorAll('[data-preset]');
    const freeformBtn = document.getElementById('freeformBtn');
    let currentPreset = 'quote'; // Initialize with quote
    let isPresetChanging = false; // Flag to track if text change is from preset button
    
    // Helper function to update active button state
    function updateActiveButton(presetType) {
        // Remove active from all buttons
        presetButtons.forEach(btn => btn.classList.remove('active'));
        
        // Add active to the specified button
        if (presetType) {
            const targetBtn = document.querySelector(`[data-preset="${presetType}"]`);
            if (targetBtn) {
                targetBtn.classList.add('active');
            }
        }
        currentPreset = presetType;
        updateQuoteModeState(presetType === 'quote');
    }

    function updateQuoteModeState(isQuote) {
        if (!demoResults) return;
        demoResults.classList.toggle('quote-mode', Boolean(isQuote));
    }
    
    // Initialize with quote button active
    updateActiveButton('quote');
    
    presetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const presetType = btn.getAttribute('data-preset');
            
            if (presetType === 'freeform') {
                // Freeform button: clear text
                isPresetChanging = true;
                demoTextInput.value = '';
                updateCharCounter();
                updateActiveButton('freeform');
                isPresetChanging = false;
            } else {
                // Other preset buttons: set text
                const text = presetTexts[presetType];
                if (text) {
                    isPresetChanging = true;
                    demoTextInput.value = text;
                    updateCharCounter();
                    updateActiveButton(presetType);
                    isPresetChanging = false;
                }
            }
        });
    });

    // Update character counter on input
    let previousTextValue = demoTextInput.value;
    demoTextInput.addEventListener('input', () => {
        updateCharCounter();
        
        // If text was modified by user (not from preset button), switch to freeform
        if (!isPresetChanging && demoTextInput.value !== previousTextValue) {
            updateActiveButton('freeform');
        }
        
        previousTextValue = demoTextInput.value;
    });
    
    // Update font size when window is resized (for responsive width-based font sizing)
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            updateCharCounter();
        }, 100);
    });
    
    // Initialize character counter
    updateCharCounter();

    // Voice toggle button handlers
    const voiceToggleTexts = document.querySelectorAll('.voice-toggle-text');
    
    // Disable voice toggle texts initially
    voiceToggleTexts.forEach(text => text.classList.add('disabled'));
    
    voiceToggleTexts.forEach(text => {
        text.addEventListener('click', async () => {
            if (text.classList.contains('disabled')) return;
            
            const selectedVoice = text.getAttribute('data-voice');
            
            // Don't reload if already selected
            if (selectedVoice === currentVoice) {
                return;
            }
            
            // Update UI
            voiceToggleTexts.forEach(t => t.classList.remove('active'));
            text.classList.add('active');
            
            // Disable all controls while loading
            const wasDisabled = demoGenerateBtn.disabled;
            demoGenerateBtn.disabled = true;
            voiceToggleTexts.forEach(t => t.classList.add('disabled'));
            
            try {
                await switchVoice(selectedVoice);
                // Re-enable texts if models are loaded
                if (models && cfgs && processors) {
                    demoGenerateBtn.disabled = false;
                    voiceToggleTexts.forEach(t => t.classList.remove('disabled'));
                }
            } catch (error) {
                console.error('Failed to switch voice:', error);
                // Revert UI on error
                voiceToggleTexts.forEach(t => t.classList.remove('active'));
                document.querySelector(`[data-voice="${currentVoice}"]`).classList.add('active');
                // Re-enable texts
                voiceToggleTexts.forEach(t => t.classList.remove('disabled'));
                if (!wasDisabled) demoGenerateBtn.disabled = false;
            }
        });
    });


    // Title animation setup
    const demoTitleLeft = document.querySelector('.demo-title-left');
    const demoTitleRight = document.querySelector('.demo-title-right');
    const demoInputSection = document.querySelector('.demo-input-section');
    const demoOutputSection = document.querySelector('.demo-output-section');

    // Initialize Text with letters wrapped in spans
    if (demoTitleLeft) {
        const text = demoTitleLeft.textContent.trim();
        demoTitleLeft.innerHTML = text.split('').map(char => 
            char === ' ' ? ' ' : `<span class="letter visible">${char}</span>`
        ).join('');
    }

    // Text animation on demo-input-section click
    if (demoInputSection && demoTitleLeft) {
        demoInputSection.addEventListener('click', () => {
            const letters = demoTitleLeft.querySelectorAll('.letter');
            // Reset all letters
            letters.forEach(letter => {
                letter.classList.remove('visible');
            });
            
            // Show letters one by one (total 0.25s = 0.125s / 2)
            letters.forEach((letter, index) => {
                setTimeout(() => {
                    letter.classList.add('visible');
                }, index * 0.0625 * 1000); // 0.0625s delay between each letter
            });
        });
    }

    // Speech animation on demo-output-section click
    if (demoOutputSection && demoTitleRight) {
        demoOutputSection.addEventListener('click', (event) => {
            if (event.target.closest('#demoGenerateBtn')) {
                return;
            }
            demoTitleRight.classList.remove('animate-speech');
            // Trigger reflow
            void demoTitleRight.offsetWidth;
            demoTitleRight.classList.add('animate-speech');
        });
    }

    function getProviderSlugForAudio(providerName) {
        if (!providerName) return null;
        const normalized = providerName.toLowerCase();
        switch (normalized) {
            case 'supertone':
            case 'supertonic':
                return 'supertone';
            case 'elevenlabs':
                return 'elevenlabs';
            case 'openai':
                return 'openai';
            case 'gemini':
                return 'gemini';
            default:
                return normalized;
        }
    }

    function updateTextHandlingPlayButtonState(cardState, isPlaying) {
        if (!cardState.playButton) return;
        cardState.playButton.classList.toggle('is-playing', isPlaying);
        cardState.playButton.setAttribute('aria-pressed', String(isPlaying));
        const action = isPlaying ? 'Pause' : 'Play';
        cardState.playButton.setAttribute('aria-label', `${action} ${cardState.sampleTitle} sample`);
    }

    function handleTextHandlingAudioEnded(cardState, audioEl) {
        if (cardState.currentAudio !== audioEl) {
            return;
        }
        cardState.isPlaying = false;
        cardState.isPaused = false;
        audioEl.currentTime = 0;
        updateTextHandlingPlayButtonState(cardState, false);
    }

    function getOrCreateTextHandlingAudio(cardState, providerSlug) {
        if (!cardState.audioElements.has(providerSlug)) {
            const audioPath = `audio/${providerSlug}_speech-${cardState.audioNumber}.mp3`;
            const audioEl = new Audio(audioPath);
            audioEl.preload = 'auto';
            audioEl.addEventListener('ended', () => handleTextHandlingAudioEnded(cardState, audioEl));
            cardState.audioElements.set(providerSlug, audioEl);
        }
        return cardState.audioElements.get(providerSlug);
    }

    function pauseTextHandlingAudio(cardState, { reset = false } = {}) {
        const audioEl = cardState.currentAudio;
        if (!audioEl) {
            cardState.isPlaying = false;
            if (reset) {
                cardState.isPaused = false;
            }
            return;
        }

        try {
            audioEl.pause();
        } catch (error) {
            console.warn('Failed to pause audio', error);
        }

        if (reset) {
            audioEl.currentTime = 0;
            cardState.currentAudio = null;
            cardState.isPaused = false;
        } else {
            cardState.isPaused = audioEl.currentTime > 0 && audioEl.currentTime < audioEl.duration;
        }

        cardState.isPlaying = false;
        updateTextHandlingPlayButtonState(cardState, false);
    }

    function playTextHandlingAudio(cardState, { restart = false } = {}) {
        const providerName = cardState.currentProvider;
        const providerSlug = getProviderSlugForAudio(providerName);
        if (!providerSlug) {
            return;
        }

        const audioEl = getOrCreateTextHandlingAudio(cardState, providerSlug);
        if (!audioEl) {
            return;
        }

        if (cardState.currentAudio && cardState.currentAudio !== audioEl) {
            pauseTextHandlingAudio(cardState, { reset: true });
        }

        cardState.currentAudio = audioEl;

        if (restart || audioEl.ended) {
            audioEl.currentTime = 0;
        }

        pauseTextHandlingPlayersExcept(cardState.playerRecord);

        const playPromise = audioEl.play();
        if (playPromise && typeof playPromise.catch === 'function') {
            playPromise.catch(error => {
                console.warn('Failed to play text-handling audio', error);
            });
        }

        cardState.isPlaying = true;
        cardState.isPaused = false;
        updateTextHandlingPlayButtonState(cardState, true);
    }

    function handleTextHandlingProviderSelection(cardState, option) {
        cardState.providerOptions.forEach(btn => {
            const isActive = btn === option;
            btn.classList.toggle('active', isActive);
            btn.setAttribute('aria-pressed', String(isActive));
        });

        const providerName = option.dataset.provider || option.textContent.trim();
        cardState.currentProvider = providerName;

        if (cardState.textModelLabel) {
            cardState.textModelLabel.textContent = getProviderLabel(providerName);
        }

        if (cardState.textModel) {
            cardState.textModel.setAttribute('data-selected-provider', providerName);
        }

        cardState.card.style.setProperty('--provider-color', getProviderColor(providerName));

        playTextHandlingAudio(cardState, { restart: true });
    }

    function handleTextHandlingPlayClick(cardState) {
        const activeOption = cardState.card.querySelector('.provider-option.active');
        if (!activeOption) {
            const defaultOption = cardState.providerOptions[0];
            if (defaultOption) {
                defaultOption.click();
            }
            return;
        }

        if (cardState.isPlaying) {
            pauseTextHandlingAudio(cardState);
            return;
        }

        if (cardState.isPaused && cardState.currentAudio) {
            pauseTextHandlingPlayersExcept(cardState.playerRecord);
            const resumePromise = cardState.currentAudio.play();
            if (resumePromise && typeof resumePromise.catch === 'function') {
                resumePromise.catch(error => console.warn('Failed to resume audio', error));
            }
            cardState.isPlaying = true;
            cardState.isPaused = false;
            updateTextHandlingPlayButtonState(cardState, true);
            return;
        }

        playTextHandlingAudio(cardState, { restart: false });
    }

    function initTextHandlingCards() {
        const cards = document.querySelectorAll('.text-handling-card');
        if (!cards.length) {
            return;
        }

        cards.forEach((card, index) => {
            const providerOptions = Array.from(card.querySelectorAll('.provider-option'));
            const textModel = card.querySelector('.text-model');
            const textModelLabel = card.querySelector('.text-model-label');
            const playButton = card.querySelector('.text-handling-player');
            const sampleTitle = card.querySelector('.text-handling-label')?.textContent?.trim() || 'sample';

            const cardState = {
                card,
                providerOptions,
                textModel,
                textModelLabel,
                playButton,
                sampleTitle,
                audioNumber: TEXT_HANDLING_CARD_AUDIO_MAP[index] || index + 1,
                audioElements: new Map(),
                currentProvider: null,
                currentAudio: null,
                isPlaying: false,
                isPaused: false,
                playerRecord: null
            };

            const playerRecord = {
                pausePlayback: () => pauseTextHandlingAudio(cardState)
            };
            cardState.playerRecord = playerRecord;
            textHandlingAudioPlayers.push(playerRecord);

            providerOptions.forEach(option => {
                option.addEventListener('click', () => handleTextHandlingProviderSelection(cardState, option));
            });

            if (playButton) {
                playButton.addEventListener('click', () => handleTextHandlingPlayClick(cardState));
                updateTextHandlingPlayButtonState(cardState, false);
            }

            card.style.setProperty('--provider-color', getProviderColor('Supertone'));
        });
    }

    function getProviderColor(provider) {
        switch (provider) {
            case 'Supertone':
            case 'supertone':
                return getComputedStyle(document.documentElement).getPropertyValue('--supertone_blue') || '#227CFF';
            case 'ElevenLabs':
                return getComputedStyle(document.documentElement).getPropertyValue('--brand-elevenlabs') || '#999999';
            case 'OpenAI':
                return getComputedStyle(document.documentElement).getPropertyValue('--brand-openai') || '#52a584';
            case 'Gemini':
                return getComputedStyle(document.documentElement).getPropertyValue('--brand-gemini') || '#887eca';
            default:
                return getComputedStyle(document.documentElement).getPropertyValue('--primary') || '#227CFF';
        }
    }

    function getProviderLabel(provider) {
        switch ((provider || '').toLowerCase()) {
            case 'supertone':
                return 'Supertonic';
            case 'elevenlabs':
                return 'Flash v2.5';
            case 'openai':
                return 'TTS-1';
            case 'gemini':
                return '2.5 Flash TTS';
            default:
                return provider || 'Supertonic';
        }
    }

    initTextHandlingCards();

    // Initialize models
    initializeModels();
})();
