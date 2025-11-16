// instrumentManager.js
class instrumentManager {
    constructor() {
        this.instruments = {};
        this.toneLoaded = false;
        this.started = false;
        this.currentTheme = "jazz";
        this.defaultBpm = 120;

    }

    // Load Tone.js from CDN if not already loaded
    async loadTone() {
        if (this.toneLoaded) return;
        await new Promise((resolve) => {
        if (window.Tone) {
            resolve();
            return;
        }
        const script = document.createElement("script");
        script.src = "https://cdn.jsdelivr.net/npm/tone@latest/build/Tone.js";
        script.async = true;
        script.onload = () => {
            this.toneLoaded = true;
            resolve();
        };
        document.body.appendChild(script);
        });
    }

    // Load and create a player
    async loadInstrument(name, url, { loop = true, volume = 0 } = {}) {
        await this.loadTone();
        await window.Tone.start(); // user gesture unlock
        if (!this.instruments[name]) {
            const player = new window.Tone.Player(url).toDestination();
            player.loop = loop;
            player.volume.value = volume;
            player.mute = true;  // start muted
            player.playbackRate = 1;
            this.instruments[name] = player;
            console.log(`Loading instrument: ${name} from ${url}`);
        }
    }

    async setTheme(theme) {
        this.currentTheme = theme;

        // Stop all instruments before switching
        Object.values(this.instruments).forEach((player) => {
            try {
                player.stop();
            } catch {}
        });

        this.started = false;
    }


    // start all instruments in sync
    _startAllOnce() {
        if (this.started) return;

        Object.values(this.instruments).forEach((player) => {
            player.start(0);   // start all at the exact same time
            player.mute = true;  // keep muted until toggled
        });

       this.started = true;
    }
    
    toggle(name) { 
        this._startAllOnce();

        const themedName = `${this.currentTheme}_${name}`; // e.g. “jazz_keyboard”

        const player = this.instruments[themedName];
        if (!player) return;

        player.mute = !player.mute;
        console.log(`${name} is now ${player.mute ? "muted" : "unmuted"}`);
    }

    
    setTempo(bpm) {
        this.currentBpm = bpm;
        const rate = bpm / this.defaultBpm;

        Object.values(this.instruments).forEach((player) => {
            player.playbackRate = rate;
        });
    }

    setVolume(name, volume) {
        const themedName = `${this.currentTheme}_${name}`; // e.g. “jazz_keyboard”
        const player = this.instruments[themedName];
        if (player) player.volume.value = volume;
    }
}

export default new instrumentManager();
