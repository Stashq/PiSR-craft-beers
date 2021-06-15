<script lang="ts">
	
	import BeerList from './Beer_image_list.svelte'
	let rec_beers;
	let best_beers;
	let popular_beers;

	import { actual_beer, beers} from './store.js';
	const x = beers.subscribe
	(value => {//console.log(value)
		rec_beers = value.recommended;
		best_beers =  value.best;
		popular_beers =  value.popular;
	})
	let row_names = ["Twoje rekomendacje","Najlepsze piwa","Najpopularniejsze"]
</script>

	<div class="imageholder" style="">
			{#each [rec_beers,best_beers,popular_beers] as beers, i}
		<div class="beer_row_title_div">
			<h2 class="beer_row_title">{row_names[i]}</h2>
		</div>
			  <BeerList beerlist={beers}></BeerList>
			{/each}
	  </div>

<style>
	.beer_row_title_div{
		display: flex;
		justify-content:center;
	}
	.beer_row_title{
		color:#e6a14e;

	}
	.beer_img{
		width: 50%;
		height: 70%;
	}
	.imageholder{
		min-width: 100px;
		width:100%;
		display: flex;
		flex-direction: column;
		justify-content: space-around;
	}
</style>