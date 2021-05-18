import { writable } from 'svelte/store';

export const actual_user = writable(0);
export const users = writable([]);

export const beers = writable({
    'best':[],
    'recommended':[],
    'popular':[],
});
export const actual_beer = writable([]);